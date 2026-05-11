# Central Depository Services (India) Ltd. (CDSL)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1261.00
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
| ENTRY1 | 50 |
| ENTRY2 | 0 |
| PARTIAL | 26 |
| TARGET_HIT | 13 |
| STOP_HIT | 37 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 76 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 39 / 37
- **Target hits / Stop hits / Partials:** 13 / 37 / 26
- **Avg / median % per leg:** 0.35% / 0.31%
- **Sum % (uncompounded):** 26.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 48 | 25 | 52.1% | 8 | 23 | 17 | 0.35% | 17.0% |
| BUY @ 2nd Alert (retest1) | 48 | 25 | 52.1% | 8 | 23 | 17 | 0.35% | 17.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 28 | 14 | 50.0% | 5 | 14 | 9 | 0.34% | 9.6% |
| SELL @ 2nd Alert (retest1) | 28 | 14 | 50.0% | 5 | 14 | 9 | 0.34% | 9.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 76 | 39 | 51.3% | 13 | 37 | 26 | 0.35% | 26.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 09:30:00 | 1060.47 | 1066.89 | 0.00 | ORB-short ORB[1062.50,1074.85] vol=2.0x ATR=3.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 09:50:00 | 1054.94 | 1064.25 | 0.00 | T1 1.5R @ 1054.94 |
| Target hit | 2024-05-16 15:20:00 | 1047.50 | 1054.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2024-05-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 09:45:00 | 1070.95 | 1061.11 | 0.00 | ORB-long ORB[1051.78,1067.18] vol=3.2x ATR=5.40 |
| Stop hit — per-position SL triggered | 2024-05-21 09:55:00 | 1065.55 | 1062.01 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 09:30:00 | 1055.22 | 1059.25 | 0.00 | ORB-short ORB[1056.15,1064.70] vol=1.8x ATR=3.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 09:35:00 | 1050.52 | 1057.48 | 0.00 | T1 1.5R @ 1050.52 |
| Target hit | 2024-05-28 15:20:00 | 1042.58 | 1049.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2024-05-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 09:35:00 | 1028.55 | 1034.48 | 0.00 | ORB-short ORB[1032.53,1041.75] vol=2.8x ATR=3.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 09:45:00 | 1023.61 | 1032.27 | 0.00 | T1 1.5R @ 1023.61 |
| Stop hit — per-position SL triggered | 2024-05-30 10:15:00 | 1028.55 | 1028.48 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 09:30:00 | 1031.40 | 1037.30 | 0.00 | ORB-short ORB[1033.00,1048.13] vol=1.8x ATR=3.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 09:45:00 | 1025.46 | 1033.23 | 0.00 | T1 1.5R @ 1025.46 |
| Stop hit — per-position SL triggered | 2024-05-31 10:05:00 | 1031.40 | 1030.68 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 09:30:00 | 1033.95 | 1025.40 | 0.00 | ORB-long ORB[1011.55,1026.00] vol=4.9x ATR=4.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-07 09:35:00 | 1040.77 | 1028.36 | 0.00 | T1 1.5R @ 1040.77 |
| Target hit | 2024-06-07 15:10:00 | 1041.10 | 1041.24 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — SELL (started 2024-06-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 09:30:00 | 1040.05 | 1045.09 | 0.00 | ORB-short ORB[1041.63,1048.97] vol=1.8x ATR=3.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 09:50:00 | 1034.24 | 1042.32 | 0.00 | T1 1.5R @ 1034.24 |
| Stop hit — per-position SL triggered | 2024-06-10 10:20:00 | 1040.05 | 1040.85 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 09:40:00 | 1043.00 | 1038.52 | 0.00 | ORB-long ORB[1030.20,1041.93] vol=1.6x ATR=4.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 12:20:00 | 1048.99 | 1043.04 | 0.00 | T1 1.5R @ 1048.99 |
| Target hit | 2024-06-11 14:45:00 | 1046.80 | 1048.04 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2024-06-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 10:05:00 | 1065.80 | 1055.63 | 0.00 | ORB-long ORB[1045.88,1056.93] vol=5.1x ATR=3.90 |
| Stop hit — per-position SL triggered | 2024-06-12 10:15:00 | 1061.90 | 1058.10 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 10:05:00 | 1060.50 | 1054.04 | 0.00 | ORB-long ORB[1050.00,1058.43] vol=3.5x ATR=3.34 |
| Stop hit — per-position SL triggered | 2024-06-13 10:10:00 | 1057.16 | 1054.62 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-14 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 10:10:00 | 1060.65 | 1056.84 | 0.00 | ORB-long ORB[1047.83,1058.85] vol=6.2x ATR=3.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 10:15:00 | 1065.69 | 1059.13 | 0.00 | T1 1.5R @ 1065.69 |
| Stop hit — per-position SL triggered | 2024-06-14 11:00:00 | 1060.65 | 1060.39 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-06-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 10:00:00 | 1032.55 | 1028.13 | 0.00 | ORB-long ORB[1018.18,1032.47] vol=1.8x ATR=2.76 |
| Stop hit — per-position SL triggered | 2024-06-21 10:05:00 | 1029.79 | 1028.32 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 11:00:00 | 1175.33 | 1166.57 | 0.00 | ORB-long ORB[1158.03,1175.28] vol=3.8x ATR=4.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 11:05:00 | 1181.80 | 1168.35 | 0.00 | T1 1.5R @ 1181.80 |
| Stop hit — per-position SL triggered | 2024-07-04 11:10:00 | 1175.33 | 1168.87 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 09:30:00 | 1167.10 | 1160.15 | 0.00 | ORB-long ORB[1149.95,1167.00] vol=3.4x ATR=4.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 09:40:00 | 1173.49 | 1166.02 | 0.00 | T1 1.5R @ 1173.49 |
| Target hit | 2024-07-11 15:20:00 | 1219.00 | 1212.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — SELL (started 2024-07-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 10:00:00 | 1143.83 | 1153.16 | 0.00 | ORB-short ORB[1149.00,1164.10] vol=1.5x ATR=4.54 |
| Stop hit — per-position SL triggered | 2024-07-19 10:25:00 | 1148.37 | 1151.75 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 09:30:00 | 1139.30 | 1149.15 | 0.00 | ORB-short ORB[1144.68,1156.45] vol=2.2x ATR=3.88 |
| Stop hit — per-position SL triggered | 2024-07-23 09:35:00 | 1143.18 | 1148.25 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 09:50:00 | 1188.47 | 1176.60 | 0.00 | ORB-long ORB[1159.50,1177.00] vol=2.2x ATR=8.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-24 10:55:00 | 1201.38 | 1186.27 | 0.00 | T1 1.5R @ 1201.38 |
| Target hit | 2024-07-24 13:25:00 | 1190.93 | 1192.04 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — BUY (started 2024-08-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 09:35:00 | 1208.90 | 1193.04 | 0.00 | ORB-long ORB[1176.05,1194.00] vol=3.6x ATR=6.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-08 09:45:00 | 1218.06 | 1203.26 | 0.00 | T1 1.5R @ 1218.06 |
| Stop hit — per-position SL triggered | 2024-08-08 09:55:00 | 1208.90 | 1204.64 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 09:30:00 | 1263.85 | 1248.17 | 0.00 | ORB-long ORB[1234.00,1252.50] vol=4.2x ATR=7.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 09:40:00 | 1275.83 | 1258.59 | 0.00 | T1 1.5R @ 1275.83 |
| Target hit | 2024-08-09 10:20:00 | 1272.93 | 1275.68 | 0.00 | Trail-exit close<VWAP |

### Cycle 20 — SELL (started 2024-08-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 09:30:00 | 1260.25 | 1274.11 | 0.00 | ORB-short ORB[1269.00,1285.38] vol=2.1x ATR=6.05 |
| Stop hit — per-position SL triggered | 2024-08-14 09:45:00 | 1266.30 | 1270.64 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-22 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 10:05:00 | 1467.30 | 1451.72 | 0.00 | ORB-long ORB[1433.73,1455.50] vol=3.8x ATR=7.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 10:15:00 | 1478.25 | 1456.13 | 0.00 | T1 1.5R @ 1478.25 |
| Stop hit — per-position SL triggered | 2024-08-22 10:20:00 | 1467.30 | 1456.57 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-09-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 09:45:00 | 1417.45 | 1426.37 | 0.00 | ORB-short ORB[1417.50,1432.00] vol=1.5x ATR=4.39 |
| Stop hit — per-position SL triggered | 2024-09-03 09:50:00 | 1421.84 | 1425.93 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-09-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-06 09:35:00 | 1431.40 | 1423.94 | 0.00 | ORB-long ORB[1417.05,1427.00] vol=3.4x ATR=4.20 |
| Stop hit — per-position SL triggered | 2024-09-06 09:45:00 | 1427.20 | 1426.71 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-09-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 09:55:00 | 1365.30 | 1375.65 | 0.00 | ORB-short ORB[1371.00,1384.45] vol=2.1x ATR=4.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 10:50:00 | 1358.19 | 1370.48 | 0.00 | T1 1.5R @ 1358.19 |
| Target hit | 2024-09-10 15:20:00 | 1354.95 | 1361.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — BUY (started 2024-09-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 09:30:00 | 1399.70 | 1389.59 | 0.00 | ORB-long ORB[1377.00,1394.00] vol=2.8x ATR=5.03 |
| Stop hit — per-position SL triggered | 2024-09-13 09:35:00 | 1394.67 | 1391.66 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-09-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 09:55:00 | 1481.80 | 1492.35 | 0.00 | ORB-short ORB[1490.20,1508.95] vol=1.6x ATR=4.80 |
| Stop hit — per-position SL triggered | 2024-09-26 10:10:00 | 1486.60 | 1491.55 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-10-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 11:10:00 | 1431.00 | 1436.76 | 0.00 | ORB-short ORB[1436.45,1454.25] vol=1.8x ATR=5.57 |
| Stop hit — per-position SL triggered | 2024-10-01 11:20:00 | 1436.57 | 1436.61 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-10-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 10:05:00 | 1497.00 | 1478.22 | 0.00 | ORB-long ORB[1469.20,1491.10] vol=1.7x ATR=6.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 10:10:00 | 1507.43 | 1482.86 | 0.00 | T1 1.5R @ 1507.43 |
| Stop hit — per-position SL triggered | 2024-10-11 10:15:00 | 1497.00 | 1485.14 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-11-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 09:35:00 | 1580.00 | 1568.40 | 0.00 | ORB-long ORB[1555.55,1572.85] vol=3.8x ATR=8.68 |
| Stop hit — per-position SL triggered | 2024-11-25 09:55:00 | 1571.32 | 1572.28 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-11-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 09:35:00 | 1614.95 | 1602.13 | 0.00 | ORB-long ORB[1583.00,1606.50] vol=3.1x ATR=7.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-27 09:45:00 | 1625.50 | 1608.67 | 0.00 | T1 1.5R @ 1625.50 |
| Stop hit — per-position SL triggered | 2024-11-27 09:55:00 | 1614.95 | 1610.89 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-11-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 09:30:00 | 1633.15 | 1616.73 | 0.00 | ORB-long ORB[1600.00,1619.95] vol=3.7x ATR=6.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 09:40:00 | 1643.56 | 1627.66 | 0.00 | T1 1.5R @ 1643.56 |
| Target hit | 2024-11-29 10:10:00 | 1638.15 | 1641.70 | 0.00 | Trail-exit close<VWAP |

### Cycle 32 — BUY (started 2024-12-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 10:25:00 | 1921.40 | 1909.63 | 0.00 | ORB-long ORB[1901.00,1921.00] vol=1.7x ATR=6.68 |
| Stop hit — per-position SL triggered | 2024-12-11 11:55:00 | 1914.72 | 1913.26 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-12-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 09:35:00 | 1911.75 | 1921.54 | 0.00 | ORB-short ORB[1914.00,1940.00] vol=1.6x ATR=6.85 |
| Stop hit — per-position SL triggered | 2024-12-13 09:40:00 | 1918.60 | 1921.06 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-12-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 09:45:00 | 1982.00 | 1970.17 | 0.00 | ORB-long ORB[1947.70,1973.15] vol=2.0x ATR=5.81 |
| Stop hit — per-position SL triggered | 2024-12-17 10:00:00 | 1976.19 | 1971.97 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-12-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 09:40:00 | 1792.00 | 1808.12 | 0.00 | ORB-short ORB[1800.00,1822.85] vol=1.9x ATR=6.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 09:45:00 | 1782.81 | 1804.33 | 0.00 | T1 1.5R @ 1782.81 |
| Stop hit — per-position SL triggered | 2024-12-26 09:55:00 | 1792.00 | 1802.33 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-12-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 09:40:00 | 1788.80 | 1779.58 | 0.00 | ORB-long ORB[1765.15,1788.00] vol=1.6x ATR=7.60 |
| Stop hit — per-position SL triggered | 2024-12-30 10:20:00 | 1781.20 | 1782.38 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-01-01 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 09:50:00 | 1783.30 | 1770.97 | 0.00 | ORB-long ORB[1752.05,1771.80] vol=2.1x ATR=6.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 09:55:00 | 1792.48 | 1775.05 | 0.00 | T1 1.5R @ 1792.48 |
| Target hit | 2025-01-01 12:55:00 | 1798.50 | 1798.54 | 0.00 | Trail-exit close<VWAP |

### Cycle 38 — BUY (started 2025-01-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 09:35:00 | 1821.75 | 1813.46 | 0.00 | ORB-long ORB[1807.00,1816.55] vol=1.5x ATR=4.73 |
| Stop hit — per-position SL triggered | 2025-01-03 10:00:00 | 1817.02 | 1818.93 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-01-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 09:30:00 | 1571.30 | 1583.44 | 0.00 | ORB-short ORB[1575.20,1598.95] vol=2.2x ATR=7.22 |
| Stop hit — per-position SL triggered | 2025-01-15 09:35:00 | 1578.52 | 1582.74 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-01-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 09:30:00 | 1579.30 | 1586.62 | 0.00 | ORB-short ORB[1582.70,1606.00] vol=2.4x ATR=4.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 09:35:00 | 1572.36 | 1584.23 | 0.00 | T1 1.5R @ 1572.36 |
| Target hit | 2025-01-21 15:20:00 | 1525.70 | 1543.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — BUY (started 2025-01-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 09:45:00 | 1506.45 | 1496.00 | 0.00 | ORB-long ORB[1485.45,1506.20] vol=1.5x ATR=6.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 09:50:00 | 1516.93 | 1500.02 | 0.00 | T1 1.5R @ 1516.93 |
| Stop hit — per-position SL triggered | 2025-01-23 09:55:00 | 1506.45 | 1502.16 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-01-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 10:10:00 | 1282.65 | 1275.52 | 0.00 | ORB-long ORB[1257.65,1276.60] vol=2.0x ATR=9.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 10:25:00 | 1296.33 | 1276.98 | 0.00 | T1 1.5R @ 1296.33 |
| Stop hit — per-position SL triggered | 2025-01-29 11:10:00 | 1282.65 | 1280.68 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-30 09:30:00 | 1287.00 | 1297.76 | 0.00 | ORB-short ORB[1292.30,1307.40] vol=2.0x ATR=6.30 |
| Stop hit — per-position SL triggered | 2025-01-30 09:35:00 | 1293.30 | 1297.30 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-01-31 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 09:55:00 | 1292.00 | 1281.14 | 0.00 | ORB-long ORB[1271.65,1285.65] vol=1.6x ATR=6.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-31 10:00:00 | 1301.10 | 1283.78 | 0.00 | T1 1.5R @ 1301.10 |
| Stop hit — per-position SL triggered | 2025-01-31 10:05:00 | 1292.00 | 1284.79 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-02-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 09:30:00 | 1337.40 | 1327.80 | 0.00 | ORB-long ORB[1315.00,1332.00] vol=2.2x ATR=4.73 |
| Stop hit — per-position SL triggered | 2025-02-01 09:35:00 | 1332.67 | 1328.56 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-03-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 09:55:00 | 1126.00 | 1117.22 | 0.00 | ORB-long ORB[1106.25,1120.95] vol=1.8x ATR=5.42 |
| Stop hit — per-position SL triggered | 2025-03-05 10:10:00 | 1120.58 | 1118.05 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-03-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:40:00 | 1091.95 | 1076.22 | 0.00 | ORB-long ORB[1062.00,1078.25] vol=2.0x ATR=5.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 10:05:00 | 1099.66 | 1083.43 | 0.00 | T1 1.5R @ 1099.66 |
| Target hit | 2025-03-18 15:20:00 | 1120.65 | 1103.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — SELL (started 2025-03-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 09:40:00 | 1193.90 | 1209.84 | 0.00 | ORB-short ORB[1208.50,1221.00] vol=3.1x ATR=6.61 |
| Stop hit — per-position SL triggered | 2025-03-26 09:50:00 | 1200.51 | 1207.62 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-04-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 09:30:00 | 1213.65 | 1203.42 | 0.00 | ORB-long ORB[1192.55,1208.20] vol=3.9x ATR=5.14 |
| Stop hit — per-position SL triggered | 2025-04-02 09:35:00 | 1208.51 | 1204.96 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-04-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:30:00 | 1358.30 | 1369.14 | 0.00 | ORB-short ORB[1365.00,1378.90] vol=2.0x ATR=5.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:35:00 | 1349.89 | 1363.71 | 0.00 | T1 1.5R @ 1349.89 |
| Target hit | 2025-04-25 13:45:00 | 1324.80 | 1322.36 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-16 09:30:00 | 1060.47 | 2024-05-16 09:50:00 | 1054.94 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-05-16 09:30:00 | 1060.47 | 2024-05-16 15:20:00 | 1047.50 | TARGET_HIT | 0.50 | 1.22% |
| BUY | retest1 | 2024-05-21 09:45:00 | 1070.95 | 2024-05-21 09:55:00 | 1065.55 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-05-28 09:30:00 | 1055.22 | 2024-05-28 09:35:00 | 1050.52 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-05-28 09:30:00 | 1055.22 | 2024-05-28 15:20:00 | 1042.58 | TARGET_HIT | 0.50 | 1.20% |
| SELL | retest1 | 2024-05-30 09:35:00 | 1028.55 | 2024-05-30 09:45:00 | 1023.61 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-05-30 09:35:00 | 1028.55 | 2024-05-30 10:15:00 | 1028.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-31 09:30:00 | 1031.40 | 2024-05-31 09:45:00 | 1025.46 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-05-31 09:30:00 | 1031.40 | 2024-05-31 10:05:00 | 1031.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-07 09:30:00 | 1033.95 | 2024-06-07 09:35:00 | 1040.77 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2024-06-07 09:30:00 | 1033.95 | 2024-06-07 15:10:00 | 1041.10 | TARGET_HIT | 0.50 | 0.69% |
| SELL | retest1 | 2024-06-10 09:30:00 | 1040.05 | 2024-06-10 09:50:00 | 1034.24 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-06-10 09:30:00 | 1040.05 | 2024-06-10 10:20:00 | 1040.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-11 09:40:00 | 1043.00 | 2024-06-11 12:20:00 | 1048.99 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-06-11 09:40:00 | 1043.00 | 2024-06-11 14:45:00 | 1046.80 | TARGET_HIT | 0.50 | 0.36% |
| BUY | retest1 | 2024-06-12 10:05:00 | 1065.80 | 2024-06-12 10:15:00 | 1061.90 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-06-13 10:05:00 | 1060.50 | 2024-06-13 10:10:00 | 1057.16 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-06-14 10:10:00 | 1060.65 | 2024-06-14 10:15:00 | 1065.69 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-06-14 10:10:00 | 1060.65 | 2024-06-14 11:00:00 | 1060.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-21 10:00:00 | 1032.55 | 2024-06-21 10:05:00 | 1029.79 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-07-04 11:00:00 | 1175.33 | 2024-07-04 11:05:00 | 1181.80 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-07-04 11:00:00 | 1175.33 | 2024-07-04 11:10:00 | 1175.33 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-11 09:30:00 | 1167.10 | 2024-07-11 09:40:00 | 1173.49 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-07-11 09:30:00 | 1167.10 | 2024-07-11 15:20:00 | 1219.00 | TARGET_HIT | 0.50 | 4.45% |
| SELL | retest1 | 2024-07-19 10:00:00 | 1143.83 | 2024-07-19 10:25:00 | 1148.37 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-07-23 09:30:00 | 1139.30 | 2024-07-23 09:35:00 | 1143.18 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-07-24 09:50:00 | 1188.47 | 2024-07-24 10:55:00 | 1201.38 | PARTIAL | 0.50 | 1.09% |
| BUY | retest1 | 2024-07-24 09:50:00 | 1188.47 | 2024-07-24 13:25:00 | 1190.93 | TARGET_HIT | 0.50 | 0.21% |
| BUY | retest1 | 2024-08-08 09:35:00 | 1208.90 | 2024-08-08 09:45:00 | 1218.06 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2024-08-08 09:35:00 | 1208.90 | 2024-08-08 09:55:00 | 1208.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-09 09:30:00 | 1263.85 | 2024-08-09 09:40:00 | 1275.83 | PARTIAL | 0.50 | 0.95% |
| BUY | retest1 | 2024-08-09 09:30:00 | 1263.85 | 2024-08-09 10:20:00 | 1272.93 | TARGET_HIT | 0.50 | 0.72% |
| SELL | retest1 | 2024-08-14 09:30:00 | 1260.25 | 2024-08-14 09:45:00 | 1266.30 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-08-22 10:05:00 | 1467.30 | 2024-08-22 10:15:00 | 1478.25 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2024-08-22 10:05:00 | 1467.30 | 2024-08-22 10:20:00 | 1467.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-03 09:45:00 | 1417.45 | 2024-09-03 09:50:00 | 1421.84 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-09-06 09:35:00 | 1431.40 | 2024-09-06 09:45:00 | 1427.20 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-09-10 09:55:00 | 1365.30 | 2024-09-10 10:50:00 | 1358.19 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-09-10 09:55:00 | 1365.30 | 2024-09-10 15:20:00 | 1354.95 | TARGET_HIT | 0.50 | 0.76% |
| BUY | retest1 | 2024-09-13 09:30:00 | 1399.70 | 2024-09-13 09:35:00 | 1394.67 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-09-26 09:55:00 | 1481.80 | 2024-09-26 10:10:00 | 1486.60 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-10-01 11:10:00 | 1431.00 | 2024-10-01 11:20:00 | 1436.57 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-10-11 10:05:00 | 1497.00 | 2024-10-11 10:10:00 | 1507.43 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2024-10-11 10:05:00 | 1497.00 | 2024-10-11 10:15:00 | 1497.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-25 09:35:00 | 1580.00 | 2024-11-25 09:55:00 | 1571.32 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2024-11-27 09:35:00 | 1614.95 | 2024-11-27 09:45:00 | 1625.50 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-11-27 09:35:00 | 1614.95 | 2024-11-27 09:55:00 | 1614.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-29 09:30:00 | 1633.15 | 2024-11-29 09:40:00 | 1643.56 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-11-29 09:30:00 | 1633.15 | 2024-11-29 10:10:00 | 1638.15 | TARGET_HIT | 0.50 | 0.31% |
| BUY | retest1 | 2024-12-11 10:25:00 | 1921.40 | 2024-12-11 11:55:00 | 1914.72 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-12-13 09:35:00 | 1911.75 | 2024-12-13 09:40:00 | 1918.60 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-12-17 09:45:00 | 1982.00 | 2024-12-17 10:00:00 | 1976.19 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-12-26 09:40:00 | 1792.00 | 2024-12-26 09:45:00 | 1782.81 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-12-26 09:40:00 | 1792.00 | 2024-12-26 09:55:00 | 1792.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-30 09:40:00 | 1788.80 | 2024-12-30 10:20:00 | 1781.20 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-01-01 09:50:00 | 1783.30 | 2025-01-01 09:55:00 | 1792.48 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-01-01 09:50:00 | 1783.30 | 2025-01-01 12:55:00 | 1798.50 | TARGET_HIT | 0.50 | 0.85% |
| BUY | retest1 | 2025-01-03 09:35:00 | 1821.75 | 2025-01-03 10:00:00 | 1817.02 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-01-15 09:30:00 | 1571.30 | 2025-01-15 09:35:00 | 1578.52 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-01-21 09:30:00 | 1579.30 | 2025-01-21 09:35:00 | 1572.36 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-01-21 09:30:00 | 1579.30 | 2025-01-21 15:20:00 | 1525.70 | TARGET_HIT | 0.50 | 3.39% |
| BUY | retest1 | 2025-01-23 09:45:00 | 1506.45 | 2025-01-23 09:50:00 | 1516.93 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2025-01-23 09:45:00 | 1506.45 | 2025-01-23 09:55:00 | 1506.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-29 10:10:00 | 1282.65 | 2025-01-29 10:25:00 | 1296.33 | PARTIAL | 0.50 | 1.07% |
| BUY | retest1 | 2025-01-29 10:10:00 | 1282.65 | 2025-01-29 11:10:00 | 1282.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-30 09:30:00 | 1287.00 | 2025-01-30 09:35:00 | 1293.30 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-01-31 09:55:00 | 1292.00 | 2025-01-31 10:00:00 | 1301.10 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2025-01-31 09:55:00 | 1292.00 | 2025-01-31 10:05:00 | 1292.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-01 09:30:00 | 1337.40 | 2025-02-01 09:35:00 | 1332.67 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-03-05 09:55:00 | 1126.00 | 2025-03-05 10:10:00 | 1120.58 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-03-18 09:40:00 | 1091.95 | 2025-03-18 10:05:00 | 1099.66 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2025-03-18 09:40:00 | 1091.95 | 2025-03-18 15:20:00 | 1120.65 | TARGET_HIT | 0.50 | 2.63% |
| SELL | retest1 | 2025-03-26 09:40:00 | 1193.90 | 2025-03-26 09:50:00 | 1200.51 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2025-04-02 09:30:00 | 1213.65 | 2025-04-02 09:35:00 | 1208.51 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-04-25 09:30:00 | 1358.30 | 2025-04-25 09:35:00 | 1349.89 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2025-04-25 09:30:00 | 1358.30 | 2025-04-25 13:45:00 | 1324.80 | TARGET_HIT | 0.50 | 2.47% |
