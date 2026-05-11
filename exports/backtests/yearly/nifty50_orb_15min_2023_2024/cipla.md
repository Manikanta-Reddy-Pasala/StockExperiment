# CIPLA (CIPLA)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-03-25 15:25:00 (53330 bars)
- **Last close:** 1248.40
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
| ENTRY1 | 86 |
| ENTRY2 | 0 |
| PARTIAL | 40 |
| TARGET_HIT | 11 |
| STOP_HIT | 75 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 126 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 51 / 75
- **Target hits / Stop hits / Partials:** 11 / 75 / 40
- **Avg / median % per leg:** 0.09% / 0.00%
- **Sum % (uncompounded):** 11.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 58 | 21 | 36.2% | 5 | 37 | 16 | 0.08% | 4.6% |
| BUY @ 2nd Alert (retest1) | 58 | 21 | 36.2% | 5 | 37 | 16 | 0.08% | 4.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 68 | 30 | 44.1% | 6 | 38 | 24 | 0.11% | 7.3% |
| SELL @ 2nd Alert (retest1) | 68 | 30 | 44.1% | 6 | 38 | 24 | 0.11% | 7.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 126 | 51 | 40.5% | 11 | 75 | 40 | 0.09% | 11.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-18 11:15:00 | 920.75 | 923.72 | 0.00 | ORB-short ORB[924.15,929.65] vol=1.9x ATR=1.53 |
| Stop hit — per-position SL triggered | 2023-05-18 11:25:00 | 922.28 | 923.53 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-26 11:00:00 | 946.80 | 945.07 | 0.00 | ORB-long ORB[940.50,946.00] vol=2.5x ATR=1.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-26 11:30:00 | 948.91 | 945.67 | 0.00 | T1 1.5R @ 948.91 |
| Target hit | 2023-05-26 15:20:00 | 950.45 | 948.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2023-06-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-13 10:05:00 | 965.00 | 962.36 | 0.00 | ORB-long ORB[957.00,964.60] vol=1.8x ATR=2.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-13 10:40:00 | 968.33 | 963.53 | 0.00 | T1 1.5R @ 968.33 |
| Target hit | 2023-06-13 15:20:00 | 982.60 | 969.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2023-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-16 11:15:00 | 996.15 | 998.49 | 0.00 | ORB-short ORB[998.30,1004.20] vol=2.0x ATR=1.65 |
| Stop hit — per-position SL triggered | 2023-06-16 11:40:00 | 997.80 | 998.33 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2023-06-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-20 10:25:00 | 1009.75 | 1013.19 | 0.00 | ORB-short ORB[1011.50,1018.80] vol=4.1x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-20 10:40:00 | 1006.85 | 1012.24 | 0.00 | T1 1.5R @ 1006.85 |
| Stop hit — per-position SL triggered | 2023-06-20 11:40:00 | 1009.75 | 1009.76 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2023-06-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-23 10:55:00 | 991.00 | 991.63 | 0.00 | ORB-short ORB[993.05,999.95] vol=3.6x ATR=1.83 |
| Stop hit — per-position SL triggered | 2023-06-23 13:35:00 | 992.83 | 991.20 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-07-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 10:05:00 | 1026.40 | 1023.40 | 0.00 | ORB-long ORB[1016.35,1024.50] vol=2.0x ATR=2.62 |
| Stop hit — per-position SL triggered | 2023-07-03 11:05:00 | 1023.78 | 1024.73 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-07-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 09:45:00 | 1018.70 | 1014.93 | 0.00 | ORB-long ORB[1011.40,1015.00] vol=1.6x ATR=1.99 |
| Stop hit — per-position SL triggered | 2023-07-06 10:55:00 | 1016.71 | 1017.57 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-07-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-10 10:35:00 | 1025.00 | 1019.91 | 0.00 | ORB-long ORB[1013.00,1024.00] vol=1.7x ATR=2.51 |
| Stop hit — per-position SL triggered | 2023-07-10 10:55:00 | 1022.49 | 1020.70 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-07-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 09:40:00 | 1028.65 | 1023.46 | 0.00 | ORB-long ORB[1018.00,1025.00] vol=1.6x ATR=2.57 |
| Stop hit — per-position SL triggered | 2023-07-11 10:10:00 | 1026.08 | 1025.32 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-07-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 11:10:00 | 1030.00 | 1024.12 | 0.00 | ORB-long ORB[1019.35,1028.00] vol=2.2x ATR=1.92 |
| Stop hit — per-position SL triggered | 2023-07-14 11:35:00 | 1028.08 | 1024.80 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-07-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-17 11:10:00 | 1029.90 | 1034.05 | 0.00 | ORB-short ORB[1031.80,1038.50] vol=3.4x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-17 12:20:00 | 1026.72 | 1032.53 | 0.00 | T1 1.5R @ 1026.72 |
| Stop hit — per-position SL triggered | 2023-07-17 12:50:00 | 1029.90 | 1032.32 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-07-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-18 10:40:00 | 1027.10 | 1030.94 | 0.00 | ORB-short ORB[1028.00,1035.20] vol=3.6x ATR=1.98 |
| Stop hit — per-position SL triggered | 2023-07-18 10:55:00 | 1029.08 | 1030.74 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-07-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-20 10:35:00 | 1043.55 | 1034.71 | 0.00 | ORB-long ORB[1027.00,1038.95] vol=2.5x ATR=2.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-20 10:55:00 | 1047.48 | 1037.07 | 0.00 | T1 1.5R @ 1047.48 |
| Stop hit — per-position SL triggered | 2023-07-20 11:00:00 | 1043.55 | 1037.28 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-07-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-24 09:55:00 | 1039.45 | 1046.17 | 0.00 | ORB-short ORB[1045.80,1051.05] vol=1.7x ATR=3.01 |
| Stop hit — per-position SL triggered | 2023-07-24 10:20:00 | 1042.46 | 1045.45 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-07-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-28 09:35:00 | 1189.90 | 1179.61 | 0.00 | ORB-long ORB[1165.85,1182.65] vol=1.7x ATR=5.95 |
| Stop hit — per-position SL triggered | 2023-07-28 09:40:00 | 1183.95 | 1180.37 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-08-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-09 11:00:00 | 1257.25 | 1262.11 | 0.00 | ORB-short ORB[1260.05,1271.65] vol=3.1x ATR=2.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-09 11:10:00 | 1253.11 | 1261.57 | 0.00 | T1 1.5R @ 1253.11 |
| Stop hit — per-position SL triggered | 2023-08-09 12:10:00 | 1257.25 | 1260.82 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-08-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-16 10:00:00 | 1240.45 | 1232.28 | 0.00 | ORB-long ORB[1226.40,1235.55] vol=2.0x ATR=2.92 |
| Stop hit — per-position SL triggered | 2023-08-16 10:35:00 | 1237.53 | 1235.85 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-08-22 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-22 10:20:00 | 1222.05 | 1226.23 | 0.00 | ORB-short ORB[1228.25,1240.25] vol=4.3x ATR=2.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-22 11:10:00 | 1218.23 | 1224.65 | 0.00 | T1 1.5R @ 1218.23 |
| Stop hit — per-position SL triggered | 2023-08-22 11:35:00 | 1222.05 | 1224.32 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-28 09:30:00 | 1229.00 | 1224.39 | 0.00 | ORB-long ORB[1211.00,1228.00] vol=2.4x ATR=3.18 |
| Stop hit — per-position SL triggered | 2023-08-28 09:40:00 | 1225.82 | 1225.18 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-09-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-04 09:50:00 | 1239.55 | 1243.59 | 0.00 | ORB-short ORB[1242.70,1255.40] vol=1.5x ATR=2.56 |
| Stop hit — per-position SL triggered | 2023-09-04 10:00:00 | 1242.11 | 1243.33 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-09-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 09:40:00 | 1262.75 | 1255.68 | 0.00 | ORB-long ORB[1242.60,1261.20] vol=1.8x ATR=3.94 |
| Stop hit — per-position SL triggered | 2023-09-06 09:55:00 | 1258.81 | 1257.81 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-09-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-12 09:30:00 | 1258.20 | 1252.30 | 0.00 | ORB-long ORB[1243.20,1253.55] vol=2.2x ATR=2.75 |
| Stop hit — per-position SL triggered | 2023-09-12 09:35:00 | 1255.45 | 1253.17 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-09-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-13 11:15:00 | 1230.80 | 1238.55 | 0.00 | ORB-short ORB[1235.10,1250.00] vol=2.0x ATR=2.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-13 11:55:00 | 1226.52 | 1234.45 | 0.00 | T1 1.5R @ 1226.52 |
| Stop hit — per-position SL triggered | 2023-09-13 12:10:00 | 1230.80 | 1233.09 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2023-09-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-15 10:55:00 | 1230.70 | 1235.73 | 0.00 | ORB-short ORB[1232.35,1239.20] vol=1.5x ATR=1.90 |
| Stop hit — per-position SL triggered | 2023-09-15 11:10:00 | 1232.60 | 1235.27 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2023-09-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-18 09:30:00 | 1255.00 | 1250.22 | 0.00 | ORB-long ORB[1237.10,1252.40] vol=2.3x ATR=3.40 |
| Stop hit — per-position SL triggered | 2023-09-18 09:55:00 | 1251.60 | 1251.69 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2023-09-21 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-21 10:25:00 | 1211.70 | 1225.28 | 0.00 | ORB-short ORB[1230.55,1240.25] vol=2.0x ATR=3.51 |
| Stop hit — per-position SL triggered | 2023-09-21 10:35:00 | 1215.21 | 1224.34 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2023-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-26 09:30:00 | 1174.00 | 1179.12 | 0.00 | ORB-short ORB[1175.60,1185.00] vol=1.7x ATR=2.56 |
| Stop hit — per-position SL triggered | 2023-09-26 09:35:00 | 1176.56 | 1178.80 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-09-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-28 11:10:00 | 1170.00 | 1176.29 | 0.00 | ORB-short ORB[1177.60,1185.00] vol=2.2x ATR=2.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-28 11:55:00 | 1166.19 | 1174.55 | 0.00 | T1 1.5R @ 1166.19 |
| Stop hit — per-position SL triggered | 2023-09-28 13:35:00 | 1170.00 | 1170.80 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2023-09-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-29 10:50:00 | 1190.75 | 1178.83 | 0.00 | ORB-long ORB[1166.75,1179.85] vol=1.7x ATR=3.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-29 11:10:00 | 1196.71 | 1182.50 | 0.00 | T1 1.5R @ 1196.71 |
| Stop hit — per-position SL triggered | 2023-09-29 12:05:00 | 1190.75 | 1186.00 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-10-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-04 11:15:00 | 1168.05 | 1175.95 | 0.00 | ORB-short ORB[1175.00,1185.00] vol=1.6x ATR=2.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-04 11:55:00 | 1164.48 | 1174.39 | 0.00 | T1 1.5R @ 1164.48 |
| Stop hit — per-position SL triggered | 2023-10-04 12:50:00 | 1168.05 | 1172.53 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-10-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-05 10:25:00 | 1163.00 | 1165.59 | 0.00 | ORB-short ORB[1163.40,1172.95] vol=6.3x ATR=3.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-05 10:35:00 | 1158.44 | 1165.09 | 0.00 | T1 1.5R @ 1158.44 |
| Stop hit — per-position SL triggered | 2023-10-05 11:05:00 | 1163.00 | 1163.30 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-10-09 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-09 10:20:00 | 1163.50 | 1162.58 | 0.00 | ORB-long ORB[1151.65,1163.15] vol=4.1x ATR=2.88 |
| Stop hit — per-position SL triggered | 2023-10-09 10:30:00 | 1160.62 | 1162.49 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 11:15:00 | 1165.80 | 1162.82 | 0.00 | ORB-long ORB[1146.50,1162.90] vol=2.5x ATR=2.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-11 11:30:00 | 1169.36 | 1163.23 | 0.00 | T1 1.5R @ 1169.36 |
| Stop hit — per-position SL triggered | 2023-10-11 11:35:00 | 1165.80 | 1163.31 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-10-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-13 10:45:00 | 1168.20 | 1164.02 | 0.00 | ORB-long ORB[1150.00,1163.10] vol=1.6x ATR=2.21 |
| Stop hit — per-position SL triggered | 2023-10-13 11:05:00 | 1165.99 | 1164.77 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-10-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 09:50:00 | 1171.50 | 1166.80 | 0.00 | ORB-long ORB[1160.65,1169.95] vol=2.0x ATR=2.18 |
| Stop hit — per-position SL triggered | 2023-10-17 10:25:00 | 1169.32 | 1168.44 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2023-10-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-27 10:55:00 | 1164.20 | 1157.23 | 0.00 | ORB-long ORB[1149.25,1161.15] vol=1.8x ATR=3.18 |
| Stop hit — per-position SL triggered | 2023-10-27 11:00:00 | 1161.02 | 1157.43 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2023-11-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-01 11:00:00 | 1191.80 | 1199.59 | 0.00 | ORB-short ORB[1200.00,1208.70] vol=1.7x ATR=2.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-01 11:25:00 | 1187.63 | 1197.99 | 0.00 | T1 1.5R @ 1187.63 |
| Stop hit — per-position SL triggered | 2023-11-01 11:50:00 | 1191.80 | 1197.08 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2023-11-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-02 09:50:00 | 1201.10 | 1207.62 | 0.00 | ORB-short ORB[1204.50,1211.85] vol=1.6x ATR=3.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-02 10:10:00 | 1196.29 | 1205.30 | 0.00 | T1 1.5R @ 1196.29 |
| Stop hit — per-position SL triggered | 2023-11-02 10:45:00 | 1201.10 | 1203.42 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2023-11-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-06 09:30:00 | 1208.45 | 1211.95 | 0.00 | ORB-short ORB[1209.20,1217.00] vol=2.2x ATR=3.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-06 10:00:00 | 1203.94 | 1209.55 | 0.00 | T1 1.5R @ 1203.94 |
| Target hit | 2023-11-06 12:35:00 | 1205.00 | 1204.76 | 0.00 | Trail-exit close>VWAP |

### Cycle 41 — BUY (started 2023-11-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-07 11:10:00 | 1213.40 | 1208.51 | 0.00 | ORB-long ORB[1200.75,1208.95] vol=4.3x ATR=2.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-07 13:05:00 | 1216.71 | 1211.39 | 0.00 | T1 1.5R @ 1216.71 |
| Stop hit — per-position SL triggered | 2023-11-07 14:20:00 | 1213.40 | 1212.98 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2023-11-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 09:30:00 | 1232.00 | 1226.18 | 0.00 | ORB-long ORB[1220.00,1227.75] vol=2.0x ATR=2.40 |
| Stop hit — per-position SL triggered | 2023-11-08 09:35:00 | 1229.60 | 1226.93 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2023-11-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 11:00:00 | 1245.80 | 1240.25 | 0.00 | ORB-long ORB[1233.85,1243.95] vol=1.7x ATR=2.37 |
| Stop hit — per-position SL triggered | 2023-11-16 11:05:00 | 1243.43 | 1240.41 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-11-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 10:40:00 | 1248.60 | 1246.19 | 0.00 | ORB-long ORB[1242.15,1247.70] vol=1.6x ATR=1.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-21 11:10:00 | 1251.46 | 1246.90 | 0.00 | T1 1.5R @ 1251.46 |
| Stop hit — per-position SL triggered | 2023-11-21 11:40:00 | 1248.60 | 1247.23 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2023-11-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-22 09:30:00 | 1273.45 | 1265.26 | 0.00 | ORB-long ORB[1254.00,1267.05] vol=3.2x ATR=2.95 |
| Stop hit — per-position SL triggered | 2023-11-22 09:35:00 | 1270.50 | 1267.78 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2023-11-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 10:35:00 | 1199.70 | 1197.55 | 0.00 | ORB-long ORB[1191.75,1199.55] vol=1.8x ATR=2.19 |
| Stop hit — per-position SL triggered | 2023-11-29 12:50:00 | 1197.51 | 1198.90 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2023-12-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-01 11:00:00 | 1208.60 | 1210.10 | 0.00 | ORB-short ORB[1209.35,1217.85] vol=2.0x ATR=2.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-01 13:55:00 | 1204.68 | 1209.10 | 0.00 | T1 1.5R @ 1204.68 |
| Target hit | 2023-12-01 15:20:00 | 1205.20 | 1208.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — BUY (started 2023-12-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-05 11:10:00 | 1222.80 | 1222.37 | 0.00 | ORB-long ORB[1216.70,1222.60] vol=7.0x ATR=2.00 |
| Stop hit — per-position SL triggered | 2023-12-05 11:20:00 | 1220.80 | 1222.32 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2023-12-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-06 11:05:00 | 1219.90 | 1222.22 | 0.00 | ORB-short ORB[1220.00,1229.85] vol=1.5x ATR=2.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-06 11:50:00 | 1216.02 | 1221.14 | 0.00 | T1 1.5R @ 1216.02 |
| Target hit | 2023-12-06 15:20:00 | 1206.75 | 1208.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — SELL (started 2023-12-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 11:00:00 | 1216.55 | 1223.90 | 0.00 | ORB-short ORB[1222.80,1231.15] vol=1.8x ATR=2.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-08 12:25:00 | 1212.12 | 1220.72 | 0.00 | T1 1.5R @ 1212.12 |
| Target hit | 2023-12-08 14:50:00 | 1216.00 | 1215.05 | 0.00 | Trail-exit close>VWAP |

### Cycle 51 — BUY (started 2023-12-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 11:05:00 | 1215.60 | 1212.74 | 0.00 | ORB-long ORB[1208.10,1215.00] vol=2.0x ATR=2.26 |
| Stop hit — per-position SL triggered | 2023-12-15 11:35:00 | 1213.34 | 1213.18 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2023-12-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-20 10:00:00 | 1233.00 | 1237.46 | 0.00 | ORB-short ORB[1234.00,1243.55] vol=2.4x ATR=3.24 |
| Stop hit — per-position SL triggered | 2023-12-20 10:05:00 | 1236.24 | 1236.96 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2023-12-22 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 10:20:00 | 1233.25 | 1232.89 | 0.00 | ORB-long ORB[1223.55,1230.95] vol=6.1x ATR=3.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-22 10:30:00 | 1238.92 | 1233.09 | 0.00 | T1 1.5R @ 1238.92 |
| Stop hit — per-position SL triggered | 2023-12-22 11:00:00 | 1233.25 | 1233.53 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2023-12-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-27 09:30:00 | 1242.55 | 1245.10 | 0.00 | ORB-short ORB[1243.05,1250.00] vol=1.6x ATR=2.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-27 09:35:00 | 1239.26 | 1244.00 | 0.00 | T1 1.5R @ 1239.26 |
| Stop hit — per-position SL triggered | 2023-12-27 09:45:00 | 1242.55 | 1243.31 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2023-12-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-29 10:50:00 | 1254.30 | 1258.82 | 0.00 | ORB-short ORB[1254.80,1267.60] vol=5.4x ATR=3.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-29 11:10:00 | 1249.70 | 1257.48 | 0.00 | T1 1.5R @ 1249.70 |
| Target hit | 2023-12-29 15:20:00 | 1246.70 | 1250.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — BUY (started 2024-01-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-02 09:40:00 | 1271.80 | 1264.93 | 0.00 | ORB-long ORB[1252.10,1267.20] vol=3.8x ATR=3.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-02 09:45:00 | 1276.92 | 1267.82 | 0.00 | T1 1.5R @ 1276.92 |
| Stop hit — per-position SL triggered | 2024-01-02 09:50:00 | 1271.80 | 1268.24 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-01-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-08 11:05:00 | 1272.85 | 1280.58 | 0.00 | ORB-short ORB[1283.00,1297.00] vol=1.6x ATR=2.46 |
| Stop hit — per-position SL triggered | 2024-01-08 11:35:00 | 1275.31 | 1279.73 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2024-01-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 11:05:00 | 1286.65 | 1278.64 | 0.00 | ORB-long ORB[1275.05,1284.00] vol=2.0x ATR=2.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-09 11:40:00 | 1290.59 | 1280.69 | 0.00 | T1 1.5R @ 1290.59 |
| Stop hit — per-position SL triggered | 2024-01-09 14:25:00 | 1286.65 | 1285.37 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-01-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-12 09:40:00 | 1311.55 | 1316.71 | 0.00 | ORB-short ORB[1315.80,1325.65] vol=2.7x ATR=3.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-12 09:55:00 | 1306.94 | 1312.53 | 0.00 | T1 1.5R @ 1306.94 |
| Stop hit — per-position SL triggered | 2024-01-12 10:05:00 | 1311.55 | 1312.20 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2024-01-15 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-15 10:30:00 | 1321.05 | 1314.79 | 0.00 | ORB-long ORB[1308.00,1317.00] vol=1.6x ATR=3.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-15 10:40:00 | 1325.83 | 1315.77 | 0.00 | T1 1.5R @ 1325.83 |
| Stop hit — per-position SL triggered | 2024-01-15 11:00:00 | 1321.05 | 1316.77 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2024-01-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-18 09:35:00 | 1270.00 | 1282.53 | 0.00 | ORB-short ORB[1283.70,1295.70] vol=3.2x ATR=4.32 |
| Stop hit — per-position SL triggered | 2024-01-18 09:40:00 | 1274.32 | 1282.03 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-01-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-20 11:00:00 | 1320.50 | 1328.45 | 0.00 | ORB-short ORB[1326.00,1337.95] vol=1.5x ATR=2.52 |
| Stop hit — per-position SL triggered | 2024-01-20 11:10:00 | 1323.02 | 1327.97 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2024-01-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-25 10:50:00 | 1388.10 | 1403.41 | 0.00 | ORB-short ORB[1404.50,1413.45] vol=1.5x ATR=3.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-25 11:10:00 | 1382.30 | 1400.44 | 0.00 | T1 1.5R @ 1382.30 |
| Stop hit — per-position SL triggered | 2024-01-25 11:20:00 | 1388.10 | 1399.13 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2024-02-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 11:00:00 | 1400.80 | 1395.64 | 0.00 | ORB-long ORB[1386.40,1394.35] vol=1.5x ATR=2.54 |
| Stop hit — per-position SL triggered | 2024-02-02 11:10:00 | 1398.26 | 1395.90 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2024-02-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 11:00:00 | 1427.70 | 1442.63 | 0.00 | ORB-short ORB[1442.05,1457.70] vol=1.7x ATR=4.81 |
| Stop hit — per-position SL triggered | 2024-02-08 13:10:00 | 1432.51 | 1436.55 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2024-02-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-13 09:35:00 | 1444.60 | 1439.85 | 0.00 | ORB-long ORB[1430.00,1443.10] vol=1.7x ATR=4.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-13 11:25:00 | 1450.76 | 1444.73 | 0.00 | T1 1.5R @ 1450.76 |
| Stop hit — per-position SL triggered | 2024-02-13 11:30:00 | 1444.60 | 1444.82 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2024-02-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-14 11:10:00 | 1426.00 | 1433.70 | 0.00 | ORB-short ORB[1435.85,1454.80] vol=1.8x ATR=3.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-14 11:30:00 | 1420.43 | 1432.00 | 0.00 | T1 1.5R @ 1420.43 |
| Stop hit — per-position SL triggered | 2024-02-14 14:55:00 | 1426.00 | 1423.75 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2024-02-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-15 09:30:00 | 1416.20 | 1420.78 | 0.00 | ORB-short ORB[1417.50,1437.95] vol=2.9x ATR=4.56 |
| Stop hit — per-position SL triggered | 2024-02-15 09:45:00 | 1420.76 | 1419.85 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2024-02-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-19 11:05:00 | 1458.35 | 1450.60 | 0.00 | ORB-long ORB[1438.95,1457.10] vol=2.9x ATR=3.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-19 12:10:00 | 1463.59 | 1456.35 | 0.00 | T1 1.5R @ 1463.59 |
| Target hit | 2024-02-19 15:20:00 | 1471.65 | 1461.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 70 — SELL (started 2024-02-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-22 10:00:00 | 1429.15 | 1437.12 | 0.00 | ORB-short ORB[1439.25,1455.55] vol=2.0x ATR=5.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-22 11:20:00 | 1420.86 | 1433.23 | 0.00 | T1 1.5R @ 1420.86 |
| Stop hit — per-position SL triggered | 2024-02-22 11:40:00 | 1429.15 | 1432.96 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2024-03-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-01 11:05:00 | 1465.60 | 1473.93 | 0.00 | ORB-short ORB[1473.90,1494.00] vol=1.8x ATR=4.28 |
| Stop hit — per-position SL triggered | 2024-03-01 11:25:00 | 1469.88 | 1473.03 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2024-03-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-04 09:35:00 | 1492.00 | 1486.60 | 0.00 | ORB-long ORB[1480.00,1489.70] vol=5.9x ATR=5.63 |
| Stop hit — per-position SL triggered | 2024-03-04 09:50:00 | 1486.37 | 1487.13 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2024-03-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-11 09:30:00 | 1506.55 | 1498.36 | 0.00 | ORB-long ORB[1488.55,1501.40] vol=1.7x ATR=4.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 09:40:00 | 1513.74 | 1505.17 | 0.00 | T1 1.5R @ 1513.74 |
| Target hit | 2024-03-11 10:30:00 | 1509.45 | 1509.51 | 0.00 | Trail-exit close<VWAP |

### Cycle 74 — SELL (started 2024-03-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-19 09:55:00 | 1447.15 | 1466.07 | 0.00 | ORB-short ORB[1469.10,1486.80] vol=1.9x ATR=4.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-19 10:25:00 | 1440.12 | 1458.17 | 0.00 | T1 1.5R @ 1440.12 |
| Stop hit — per-position SL triggered | 2024-03-19 10:30:00 | 1447.15 | 1457.48 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2024-03-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 10:55:00 | 1447.35 | 1434.44 | 0.00 | ORB-long ORB[1420.20,1431.10] vol=2.0x ATR=3.53 |
| Stop hit — per-position SL triggered | 2024-03-21 14:00:00 | 1443.82 | 1443.86 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-04-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 10:50:00 | 1449.55 | 1463.08 | 0.00 | ORB-short ORB[1465.00,1481.20] vol=2.1x ATR=4.29 |
| Stop hit — per-position SL triggered | 2024-04-04 11:15:00 | 1453.84 | 1460.64 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2024-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-08 11:15:00 | 1467.00 | 1460.92 | 0.00 | ORB-long ORB[1449.40,1459.70] vol=1.7x ATR=2.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-08 11:30:00 | 1470.78 | 1463.85 | 0.00 | T1 1.5R @ 1470.78 |
| Stop hit — per-position SL triggered | 2024-04-08 12:25:00 | 1467.00 | 1465.47 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2024-04-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-10 10:10:00 | 1435.20 | 1441.51 | 0.00 | ORB-short ORB[1446.40,1455.30] vol=1.7x ATR=3.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-10 10:45:00 | 1429.32 | 1437.90 | 0.00 | T1 1.5R @ 1429.32 |
| Stop hit — per-position SL triggered | 2024-04-10 11:05:00 | 1435.20 | 1436.89 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2024-04-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-12 11:05:00 | 1406.85 | 1414.24 | 0.00 | ORB-short ORB[1408.80,1424.85] vol=4.7x ATR=3.60 |
| Stop hit — per-position SL triggered | 2024-04-12 11:20:00 | 1410.45 | 1413.81 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2024-04-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-18 09:35:00 | 1368.25 | 1374.92 | 0.00 | ORB-short ORB[1373.15,1382.75] vol=2.0x ATR=3.38 |
| Stop hit — per-position SL triggered | 2024-04-18 09:55:00 | 1371.63 | 1373.17 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2024-04-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-23 11:00:00 | 1356.25 | 1359.66 | 0.00 | ORB-short ORB[1357.05,1369.75] vol=1.8x ATR=2.46 |
| Stop hit — per-position SL triggered | 2024-04-23 11:10:00 | 1358.71 | 1359.54 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2024-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 09:30:00 | 1371.90 | 1364.86 | 0.00 | ORB-long ORB[1352.20,1368.55] vol=2.8x ATR=3.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-24 09:35:00 | 1377.60 | 1369.19 | 0.00 | T1 1.5R @ 1377.60 |
| Target hit | 2024-04-24 15:20:00 | 1398.65 | 1386.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 83 — SELL (started 2024-04-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-29 09:35:00 | 1399.90 | 1404.71 | 0.00 | ORB-short ORB[1402.25,1420.10] vol=4.2x ATR=4.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-29 09:55:00 | 1392.97 | 1402.36 | 0.00 | T1 1.5R @ 1392.97 |
| Stop hit — per-position SL triggered | 2024-04-29 10:25:00 | 1399.90 | 1401.29 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2024-05-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-03 10:50:00 | 1432.10 | 1423.80 | 0.00 | ORB-long ORB[1415.20,1428.25] vol=2.1x ATR=3.56 |
| Stop hit — per-position SL triggered | 2024-05-03 11:05:00 | 1428.54 | 1425.15 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2024-05-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-07 10:35:00 | 1402.45 | 1412.81 | 0.00 | ORB-short ORB[1411.00,1428.85] vol=1.7x ATR=3.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 10:50:00 | 1396.91 | 1410.61 | 0.00 | T1 1.5R @ 1396.91 |
| Target hit | 2024-05-07 15:20:00 | 1389.95 | 1390.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 86 — BUY (started 2024-05-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-10 11:00:00 | 1375.40 | 1366.06 | 0.00 | ORB-long ORB[1360.90,1372.65] vol=2.6x ATR=5.28 |
| Stop hit — per-position SL triggered | 2024-05-10 11:05:00 | 1370.12 | 1367.44 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-05-18 11:15:00 | 920.75 | 2023-05-18 11:25:00 | 922.28 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-05-26 11:00:00 | 946.80 | 2023-05-26 11:30:00 | 948.91 | PARTIAL | 0.50 | 0.22% |
| BUY | retest1 | 2023-05-26 11:00:00 | 946.80 | 2023-05-26 15:20:00 | 950.45 | TARGET_HIT | 0.50 | 0.39% |
| BUY | retest1 | 2023-06-13 10:05:00 | 965.00 | 2023-06-13 10:40:00 | 968.33 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-06-13 10:05:00 | 965.00 | 2023-06-13 15:20:00 | 982.60 | TARGET_HIT | 0.50 | 1.82% |
| SELL | retest1 | 2023-06-16 11:15:00 | 996.15 | 2023-06-16 11:40:00 | 997.80 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-06-20 10:25:00 | 1009.75 | 2023-06-20 10:40:00 | 1006.85 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2023-06-20 10:25:00 | 1009.75 | 2023-06-20 11:40:00 | 1009.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-23 10:55:00 | 991.00 | 2023-06-23 13:35:00 | 992.83 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-07-03 10:05:00 | 1026.40 | 2023-07-03 11:05:00 | 1023.78 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-07-06 09:45:00 | 1018.70 | 2023-07-06 10:55:00 | 1016.71 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-07-10 10:35:00 | 1025.00 | 2023-07-10 10:55:00 | 1022.49 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-07-11 09:40:00 | 1028.65 | 2023-07-11 10:10:00 | 1026.08 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-07-14 11:10:00 | 1030.00 | 2023-07-14 11:35:00 | 1028.08 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-07-17 11:10:00 | 1029.90 | 2023-07-17 12:20:00 | 1026.72 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2023-07-17 11:10:00 | 1029.90 | 2023-07-17 12:50:00 | 1029.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-18 10:40:00 | 1027.10 | 2023-07-18 10:55:00 | 1029.08 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-07-20 10:35:00 | 1043.55 | 2023-07-20 10:55:00 | 1047.48 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-07-20 10:35:00 | 1043.55 | 2023-07-20 11:00:00 | 1043.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-24 09:55:00 | 1039.45 | 2023-07-24 10:20:00 | 1042.46 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-07-28 09:35:00 | 1189.90 | 2023-07-28 09:40:00 | 1183.95 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2023-08-09 11:00:00 | 1257.25 | 2023-08-09 11:10:00 | 1253.11 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2023-08-09 11:00:00 | 1257.25 | 2023-08-09 12:10:00 | 1257.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-16 10:00:00 | 1240.45 | 2023-08-16 10:35:00 | 1237.53 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-08-22 10:20:00 | 1222.05 | 2023-08-22 11:10:00 | 1218.23 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2023-08-22 10:20:00 | 1222.05 | 2023-08-22 11:35:00 | 1222.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-28 09:30:00 | 1229.00 | 2023-08-28 09:40:00 | 1225.82 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-09-04 09:50:00 | 1239.55 | 2023-09-04 10:00:00 | 1242.11 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-09-06 09:40:00 | 1262.75 | 2023-09-06 09:55:00 | 1258.81 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-09-12 09:30:00 | 1258.20 | 2023-09-12 09:35:00 | 1255.45 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-09-13 11:15:00 | 1230.80 | 2023-09-13 11:55:00 | 1226.52 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2023-09-13 11:15:00 | 1230.80 | 2023-09-13 12:10:00 | 1230.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-15 10:55:00 | 1230.70 | 2023-09-15 11:10:00 | 1232.60 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2023-09-18 09:30:00 | 1255.00 | 2023-09-18 09:55:00 | 1251.60 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-09-21 10:25:00 | 1211.70 | 2023-09-21 10:35:00 | 1215.21 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-09-26 09:30:00 | 1174.00 | 2023-09-26 09:35:00 | 1176.56 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-09-28 11:10:00 | 1170.00 | 2023-09-28 11:55:00 | 1166.19 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2023-09-28 11:10:00 | 1170.00 | 2023-09-28 13:35:00 | 1170.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-29 10:50:00 | 1190.75 | 2023-09-29 11:10:00 | 1196.71 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2023-09-29 10:50:00 | 1190.75 | 2023-09-29 12:05:00 | 1190.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-04 11:15:00 | 1168.05 | 2023-10-04 11:55:00 | 1164.48 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2023-10-04 11:15:00 | 1168.05 | 2023-10-04 12:50:00 | 1168.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-05 10:25:00 | 1163.00 | 2023-10-05 10:35:00 | 1158.44 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-10-05 10:25:00 | 1163.00 | 2023-10-05 11:05:00 | 1163.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-09 10:20:00 | 1163.50 | 2023-10-09 10:30:00 | 1160.62 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-10-11 11:15:00 | 1165.80 | 2023-10-11 11:30:00 | 1169.36 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-10-11 11:15:00 | 1165.80 | 2023-10-11 11:35:00 | 1165.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-13 10:45:00 | 1168.20 | 2023-10-13 11:05:00 | 1165.99 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-10-17 09:50:00 | 1171.50 | 2023-10-17 10:25:00 | 1169.32 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-10-27 10:55:00 | 1164.20 | 2023-10-27 11:00:00 | 1161.02 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-11-01 11:00:00 | 1191.80 | 2023-11-01 11:25:00 | 1187.63 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2023-11-01 11:00:00 | 1191.80 | 2023-11-01 11:50:00 | 1191.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-02 09:50:00 | 1201.10 | 2023-11-02 10:10:00 | 1196.29 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2023-11-02 09:50:00 | 1201.10 | 2023-11-02 10:45:00 | 1201.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-06 09:30:00 | 1208.45 | 2023-11-06 10:00:00 | 1203.94 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2023-11-06 09:30:00 | 1208.45 | 2023-11-06 12:35:00 | 1205.00 | TARGET_HIT | 0.50 | 0.29% |
| BUY | retest1 | 2023-11-07 11:10:00 | 1213.40 | 2023-11-07 13:05:00 | 1216.71 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2023-11-07 11:10:00 | 1213.40 | 2023-11-07 14:20:00 | 1213.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-08 09:30:00 | 1232.00 | 2023-11-08 09:35:00 | 1229.60 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-11-16 11:00:00 | 1245.80 | 2023-11-16 11:05:00 | 1243.43 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-11-21 10:40:00 | 1248.60 | 2023-11-21 11:10:00 | 1251.46 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2023-11-21 10:40:00 | 1248.60 | 2023-11-21 11:40:00 | 1248.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-22 09:30:00 | 1273.45 | 2023-11-22 09:35:00 | 1270.50 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-11-29 10:35:00 | 1199.70 | 2023-11-29 12:50:00 | 1197.51 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-12-01 11:00:00 | 1208.60 | 2023-12-01 13:55:00 | 1204.68 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2023-12-01 11:00:00 | 1208.60 | 2023-12-01 15:20:00 | 1205.20 | TARGET_HIT | 0.50 | 0.28% |
| BUY | retest1 | 2023-12-05 11:10:00 | 1222.80 | 2023-12-05 11:20:00 | 1220.80 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-12-06 11:05:00 | 1219.90 | 2023-12-06 11:50:00 | 1216.02 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2023-12-06 11:05:00 | 1219.90 | 2023-12-06 15:20:00 | 1206.75 | TARGET_HIT | 0.50 | 1.08% |
| SELL | retest1 | 2023-12-08 11:00:00 | 1216.55 | 2023-12-08 12:25:00 | 1212.12 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2023-12-08 11:00:00 | 1216.55 | 2023-12-08 14:50:00 | 1216.00 | TARGET_HIT | 0.50 | 0.05% |
| BUY | retest1 | 2023-12-15 11:05:00 | 1215.60 | 2023-12-15 11:35:00 | 1213.34 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-12-20 10:00:00 | 1233.00 | 2023-12-20 10:05:00 | 1236.24 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-12-22 10:20:00 | 1233.25 | 2023-12-22 10:30:00 | 1238.92 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2023-12-22 10:20:00 | 1233.25 | 2023-12-22 11:00:00 | 1233.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-27 09:30:00 | 1242.55 | 2023-12-27 09:35:00 | 1239.26 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2023-12-27 09:30:00 | 1242.55 | 2023-12-27 09:45:00 | 1242.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-29 10:50:00 | 1254.30 | 2023-12-29 11:10:00 | 1249.70 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2023-12-29 10:50:00 | 1254.30 | 2023-12-29 15:20:00 | 1246.70 | TARGET_HIT | 0.50 | 0.61% |
| BUY | retest1 | 2024-01-02 09:40:00 | 1271.80 | 2024-01-02 09:45:00 | 1276.92 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-01-02 09:40:00 | 1271.80 | 2024-01-02 09:50:00 | 1271.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-08 11:05:00 | 1272.85 | 2024-01-08 11:35:00 | 1275.31 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-01-09 11:05:00 | 1286.65 | 2024-01-09 11:40:00 | 1290.59 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2024-01-09 11:05:00 | 1286.65 | 2024-01-09 14:25:00 | 1286.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-12 09:40:00 | 1311.55 | 2024-01-12 09:55:00 | 1306.94 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-01-12 09:40:00 | 1311.55 | 2024-01-12 10:05:00 | 1311.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-15 10:30:00 | 1321.05 | 2024-01-15 10:40:00 | 1325.83 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-01-15 10:30:00 | 1321.05 | 2024-01-15 11:00:00 | 1321.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-18 09:35:00 | 1270.00 | 2024-01-18 09:40:00 | 1274.32 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-01-20 11:00:00 | 1320.50 | 2024-01-20 11:10:00 | 1323.02 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-01-25 10:50:00 | 1388.10 | 2024-01-25 11:10:00 | 1382.30 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-01-25 10:50:00 | 1388.10 | 2024-01-25 11:20:00 | 1388.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-02 11:00:00 | 1400.80 | 2024-02-02 11:10:00 | 1398.26 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-02-08 11:00:00 | 1427.70 | 2024-02-08 13:10:00 | 1432.51 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-02-13 09:35:00 | 1444.60 | 2024-02-13 11:25:00 | 1450.76 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-02-13 09:35:00 | 1444.60 | 2024-02-13 11:30:00 | 1444.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-14 11:10:00 | 1426.00 | 2024-02-14 11:30:00 | 1420.43 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-02-14 11:10:00 | 1426.00 | 2024-02-14 14:55:00 | 1426.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-15 09:30:00 | 1416.20 | 2024-02-15 09:45:00 | 1420.76 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-02-19 11:05:00 | 1458.35 | 2024-02-19 12:10:00 | 1463.59 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-02-19 11:05:00 | 1458.35 | 2024-02-19 15:20:00 | 1471.65 | TARGET_HIT | 0.50 | 0.91% |
| SELL | retest1 | 2024-02-22 10:00:00 | 1429.15 | 2024-02-22 11:20:00 | 1420.86 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-02-22 10:00:00 | 1429.15 | 2024-02-22 11:40:00 | 1429.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-01 11:05:00 | 1465.60 | 2024-03-01 11:25:00 | 1469.88 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-03-04 09:35:00 | 1492.00 | 2024-03-04 09:50:00 | 1486.37 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-03-11 09:30:00 | 1506.55 | 2024-03-11 09:40:00 | 1513.74 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-03-11 09:30:00 | 1506.55 | 2024-03-11 10:30:00 | 1509.45 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2024-03-19 09:55:00 | 1447.15 | 2024-03-19 10:25:00 | 1440.12 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-03-19 09:55:00 | 1447.15 | 2024-03-19 10:30:00 | 1447.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-21 10:55:00 | 1447.35 | 2024-03-21 14:00:00 | 1443.82 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-04-04 10:50:00 | 1449.55 | 2024-04-04 11:15:00 | 1453.84 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-04-08 11:15:00 | 1467.00 | 2024-04-08 11:30:00 | 1470.78 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2024-04-08 11:15:00 | 1467.00 | 2024-04-08 12:25:00 | 1467.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-10 10:10:00 | 1435.20 | 2024-04-10 10:45:00 | 1429.32 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-04-10 10:10:00 | 1435.20 | 2024-04-10 11:05:00 | 1435.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-12 11:05:00 | 1406.85 | 2024-04-12 11:20:00 | 1410.45 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-04-18 09:35:00 | 1368.25 | 2024-04-18 09:55:00 | 1371.63 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-04-23 11:00:00 | 1356.25 | 2024-04-23 11:10:00 | 1358.71 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-04-24 09:30:00 | 1371.90 | 2024-04-24 09:35:00 | 1377.60 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-04-24 09:30:00 | 1371.90 | 2024-04-24 15:20:00 | 1398.65 | TARGET_HIT | 0.50 | 1.95% |
| SELL | retest1 | 2024-04-29 09:35:00 | 1399.90 | 2024-04-29 09:55:00 | 1392.97 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-04-29 09:35:00 | 1399.90 | 2024-04-29 10:25:00 | 1399.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-03 10:50:00 | 1432.10 | 2024-05-03 11:05:00 | 1428.54 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-05-07 10:35:00 | 1402.45 | 2024-05-07 10:50:00 | 1396.91 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-05-07 10:35:00 | 1402.45 | 2024-05-07 15:20:00 | 1389.95 | TARGET_HIT | 0.50 | 0.89% |
| BUY | retest1 | 2024-05-10 11:00:00 | 1375.40 | 2024-05-10 11:05:00 | 1370.12 | STOP_HIT | 1.00 | -0.38% |
