# Coforge Ltd. (COFORGE)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1365.20
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
| ENTRY1 | 66 |
| ENTRY2 | 0 |
| PARTIAL | 22 |
| TARGET_HIT | 12 |
| STOP_HIT | 54 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 88 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 34 / 54
- **Target hits / Stop hits / Partials:** 12 / 54 / 22
- **Avg / median % per leg:** 0.13% / 0.00%
- **Sum % (uncompounded):** 11.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 63 | 28 | 44.4% | 10 | 35 | 18 | 0.15% | 9.7% |
| BUY @ 2nd Alert (retest1) | 63 | 28 | 44.4% | 10 | 35 | 18 | 0.15% | 9.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 25 | 6 | 24.0% | 2 | 19 | 4 | 0.08% | 1.9% |
| SELL @ 2nd Alert (retest1) | 25 | 6 | 24.0% | 2 | 19 | 4 | 0.08% | 1.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 88 | 34 | 38.6% | 12 | 54 | 22 | 0.13% | 11.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 11:00:00 | 920.77 | 913.29 | 0.00 | ORB-long ORB[908.20,916.29] vol=2.0x ATR=2.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 12:10:00 | 924.82 | 916.01 | 0.00 | T1 1.5R @ 924.82 |
| Stop hit — per-position SL triggered | 2024-05-15 12:45:00 | 920.77 | 916.58 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 11:15:00 | 943.02 | 937.84 | 0.00 | ORB-long ORB[934.26,943.00] vol=6.1x ATR=3.12 |
| Stop hit — per-position SL triggered | 2024-05-17 11:30:00 | 939.90 | 938.07 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 09:40:00 | 954.81 | 949.78 | 0.00 | ORB-long ORB[945.60,951.98] vol=2.2x ATR=3.45 |
| Stop hit — per-position SL triggered | 2024-05-21 09:50:00 | 951.36 | 950.56 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 09:35:00 | 995.88 | 991.55 | 0.00 | ORB-long ORB[986.17,993.63] vol=2.0x ATR=3.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 09:45:00 | 1001.49 | 995.91 | 0.00 | T1 1.5R @ 1001.49 |
| Target hit | 2024-05-23 13:15:00 | 1016.40 | 1017.67 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — BUY (started 2024-05-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 09:30:00 | 1038.46 | 1033.63 | 0.00 | ORB-long ORB[1023.81,1037.47] vol=1.8x ATR=3.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 09:35:00 | 1042.97 | 1035.47 | 0.00 | T1 1.5R @ 1042.97 |
| Stop hit — per-position SL triggered | 2024-05-27 09:40:00 | 1038.46 | 1036.22 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 09:30:00 | 1011.51 | 1019.50 | 0.00 | ORB-short ORB[1017.20,1028.21] vol=2.1x ATR=2.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 09:50:00 | 1007.38 | 1015.41 | 0.00 | T1 1.5R @ 1007.38 |
| Target hit | 2024-05-30 15:20:00 | 997.30 | 1006.62 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2024-05-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 11:10:00 | 992.70 | 994.87 | 0.00 | ORB-short ORB[994.00,1003.78] vol=2.0x ATR=2.74 |
| Stop hit — per-position SL triggered | 2024-05-31 11:35:00 | 995.44 | 994.47 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-18 10:15:00 | 1032.84 | 1040.21 | 0.00 | ORB-short ORB[1041.00,1047.68] vol=1.6x ATR=2.58 |
| Stop hit — per-position SL triggered | 2024-06-18 10:30:00 | 1035.42 | 1039.46 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 10:45:00 | 1084.85 | 1079.09 | 0.00 | ORB-long ORB[1066.70,1078.39] vol=2.4x ATR=4.04 |
| Stop hit — per-position SL triggered | 2024-06-24 10:55:00 | 1080.81 | 1079.72 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 10:45:00 | 1076.21 | 1069.77 | 0.00 | ORB-long ORB[1065.12,1074.27] vol=2.7x ATR=2.78 |
| Stop hit — per-position SL triggered | 2024-06-25 11:10:00 | 1073.43 | 1070.83 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:45:00 | 1107.28 | 1098.26 | 0.00 | ORB-long ORB[1085.15,1098.59] vol=2.0x ATR=5.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 10:55:00 | 1114.85 | 1104.15 | 0.00 | T1 1.5R @ 1114.85 |
| Stop hit — per-position SL triggered | 2024-07-01 12:15:00 | 1107.28 | 1106.95 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 09:50:00 | 1118.24 | 1111.33 | 0.00 | ORB-long ORB[1105.09,1114.40] vol=2.4x ATR=3.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 10:10:00 | 1124.16 | 1116.56 | 0.00 | T1 1.5R @ 1124.16 |
| Target hit | 2024-07-02 12:35:00 | 1126.55 | 1132.09 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — BUY (started 2024-07-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 09:30:00 | 1193.00 | 1185.61 | 0.00 | ORB-long ORB[1175.01,1190.00] vol=3.5x ATR=4.09 |
| Stop hit — per-position SL triggered | 2024-07-08 09:35:00 | 1188.91 | 1186.25 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 09:45:00 | 1141.69 | 1147.29 | 0.00 | ORB-short ORB[1147.00,1159.00] vol=2.4x ATR=3.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:05:00 | 1136.55 | 1145.90 | 0.00 | T1 1.5R @ 1136.55 |
| Stop hit — per-position SL triggered | 2024-07-10 11:40:00 | 1141.69 | 1140.74 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-29 09:45:00 | 1263.42 | 1270.56 | 0.00 | ORB-short ORB[1265.11,1282.48] vol=1.8x ATR=4.00 |
| Stop hit — per-position SL triggered | 2024-07-29 10:45:00 | 1267.42 | 1267.93 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-30 10:45:00 | 1256.42 | 1258.13 | 0.00 | ORB-short ORB[1256.78,1265.40] vol=3.6x ATR=2.88 |
| Stop hit — per-position SL triggered | 2024-07-30 10:55:00 | 1259.30 | 1258.12 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:40:00 | 1268.19 | 1261.03 | 0.00 | ORB-long ORB[1253.74,1264.19] vol=2.7x ATR=3.82 |
| Stop hit — per-position SL triggered | 2024-07-31 10:25:00 | 1264.37 | 1264.42 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-08-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-02 10:50:00 | 1229.57 | 1230.39 | 0.00 | ORB-short ORB[1233.06,1245.98] vol=1.6x ATR=3.84 |
| Stop hit — per-position SL triggered | 2024-08-02 11:40:00 | 1233.41 | 1230.63 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-07 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:25:00 | 1196.01 | 1190.78 | 0.00 | ORB-long ORB[1182.00,1194.58] vol=2.7x ATR=3.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-07 10:50:00 | 1201.75 | 1191.86 | 0.00 | T1 1.5R @ 1201.75 |
| Stop hit — per-position SL triggered | 2024-08-07 11:10:00 | 1196.01 | 1192.95 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-08-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-08 09:35:00 | 1189.79 | 1193.59 | 0.00 | ORB-short ORB[1190.20,1203.74] vol=1.6x ATR=3.22 |
| Stop hit — per-position SL triggered | 2024-08-08 09:55:00 | 1193.01 | 1193.25 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 09:35:00 | 1178.79 | 1176.77 | 0.00 | ORB-long ORB[1169.49,1177.99] vol=2.6x ATR=3.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 09:40:00 | 1183.42 | 1178.25 | 0.00 | T1 1.5R @ 1183.42 |
| Target hit | 2024-08-13 11:30:00 | 1180.39 | 1182.21 | 0.00 | Trail-exit close<VWAP |

### Cycle 22 — SELL (started 2024-08-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 09:30:00 | 1161.19 | 1168.18 | 0.00 | ORB-short ORB[1166.05,1178.00] vol=1.5x ATR=3.50 |
| Stop hit — per-position SL triggered | 2024-08-14 09:45:00 | 1164.69 | 1165.02 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 09:35:00 | 1228.00 | 1222.07 | 0.00 | ORB-long ORB[1212.41,1224.98] vol=1.6x ATR=3.70 |
| Stop hit — per-position SL triggered | 2024-08-19 09:45:00 | 1224.30 | 1223.17 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 09:50:00 | 1230.60 | 1224.95 | 0.00 | ORB-long ORB[1216.18,1228.20] vol=1.8x ATR=3.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-28 10:10:00 | 1236.26 | 1227.28 | 0.00 | T1 1.5R @ 1236.26 |
| Target hit | 2024-08-28 15:15:00 | 1249.00 | 1250.33 | 0.00 | Trail-exit close<VWAP |

### Cycle 25 — BUY (started 2024-08-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 11:05:00 | 1268.18 | 1251.46 | 0.00 | ORB-long ORB[1240.39,1254.59] vol=2.8x ATR=4.23 |
| Stop hit — per-position SL triggered | 2024-08-29 11:20:00 | 1263.95 | 1254.22 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-09-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-02 09:35:00 | 1280.81 | 1272.27 | 0.00 | ORB-long ORB[1260.82,1273.96] vol=2.5x ATR=4.21 |
| Stop hit — per-position SL triggered | 2024-09-02 09:50:00 | 1276.60 | 1275.28 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-09-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 10:20:00 | 1276.19 | 1270.84 | 0.00 | ORB-long ORB[1268.00,1275.40] vol=1.8x ATR=3.11 |
| Stop hit — per-position SL triggered | 2024-09-05 10:35:00 | 1273.08 | 1271.48 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 11:00:00 | 1370.06 | 1363.70 | 0.00 | ORB-long ORB[1355.40,1369.69] vol=2.1x ATR=3.43 |
| Stop hit — per-position SL triggered | 2024-09-11 11:10:00 | 1366.63 | 1364.11 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-09-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-13 09:35:00 | 1368.75 | 1376.92 | 0.00 | ORB-short ORB[1372.47,1385.73] vol=2.2x ATR=3.85 |
| Stop hit — per-position SL triggered | 2024-09-13 09:55:00 | 1372.60 | 1372.89 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-09-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 09:45:00 | 1392.22 | 1403.48 | 0.00 | ORB-short ORB[1400.00,1413.62] vol=1.7x ATR=5.27 |
| Stop hit — per-position SL triggered | 2024-09-16 10:55:00 | 1397.49 | 1399.22 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 09:30:00 | 1375.47 | 1385.02 | 0.00 | ORB-short ORB[1380.85,1400.00] vol=2.2x ATR=4.24 |
| Stop hit — per-position SL triggered | 2024-09-18 09:40:00 | 1379.71 | 1381.61 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 11:00:00 | 1379.50 | 1377.29 | 0.00 | ORB-long ORB[1368.30,1377.80] vol=5.3x ATR=3.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 11:05:00 | 1384.89 | 1377.57 | 0.00 | T1 1.5R @ 1384.89 |
| Stop hit — per-position SL triggered | 2024-09-24 11:25:00 | 1379.50 | 1377.83 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 09:45:00 | 1411.88 | 1402.80 | 0.00 | ORB-long ORB[1393.55,1405.99] vol=2.0x ATR=4.15 |
| Stop hit — per-position SL triggered | 2024-09-25 10:05:00 | 1407.73 | 1405.85 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-09-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-30 10:05:00 | 1398.51 | 1389.45 | 0.00 | ORB-long ORB[1374.99,1388.28] vol=1.7x ATR=5.74 |
| Stop hit — per-position SL triggered | 2024-09-30 10:20:00 | 1392.77 | 1390.31 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-10-01 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 10:05:00 | 1416.00 | 1410.94 | 0.00 | ORB-long ORB[1396.01,1412.71] vol=4.2x ATR=4.74 |
| Stop hit — per-position SL triggered | 2024-10-01 10:20:00 | 1411.26 | 1411.48 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-10-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-07 09:35:00 | 1452.62 | 1444.83 | 0.00 | ORB-long ORB[1431.70,1448.40] vol=2.5x ATR=6.06 |
| Stop hit — per-position SL triggered | 2024-10-07 09:45:00 | 1446.56 | 1446.44 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-10-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 09:35:00 | 1513.44 | 1505.97 | 0.00 | ORB-long ORB[1494.61,1513.23] vol=2.3x ATR=5.74 |
| Stop hit — per-position SL triggered | 2024-10-15 09:40:00 | 1507.70 | 1506.14 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-28 11:15:00 | 1519.85 | 1543.34 | 0.00 | ORB-short ORB[1543.10,1557.80] vol=1.6x ATR=4.96 |
| Stop hit — per-position SL triggered | 2024-10-28 11:20:00 | 1524.81 | 1542.64 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-11-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-04 09:30:00 | 1523.74 | 1515.01 | 0.00 | ORB-long ORB[1503.74,1521.00] vol=1.8x ATR=7.03 |
| Stop hit — per-position SL triggered | 2024-11-04 09:40:00 | 1516.71 | 1516.13 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-11-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 09:40:00 | 1533.76 | 1527.68 | 0.00 | ORB-long ORB[1520.42,1533.42] vol=2.0x ATR=6.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 09:50:00 | 1543.24 | 1530.73 | 0.00 | T1 1.5R @ 1543.24 |
| Target hit | 2024-11-06 15:20:00 | 1569.87 | 1556.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — SELL (started 2024-11-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 10:05:00 | 1556.39 | 1570.02 | 0.00 | ORB-short ORB[1568.58,1590.00] vol=1.6x ATR=6.21 |
| Stop hit — per-position SL triggered | 2024-11-07 11:30:00 | 1562.60 | 1563.96 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-11-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-08 09:50:00 | 1601.97 | 1584.07 | 0.00 | ORB-long ORB[1570.36,1590.88] vol=1.9x ATR=5.72 |
| Stop hit — per-position SL triggered | 2024-11-08 10:10:00 | 1596.25 | 1587.78 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-11-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 10:00:00 | 1605.30 | 1588.34 | 0.00 | ORB-long ORB[1572.58,1590.56] vol=1.8x ATR=5.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 12:25:00 | 1613.60 | 1600.42 | 0.00 | T1 1.5R @ 1613.60 |
| Target hit | 2024-11-11 15:20:00 | 1612.23 | 1604.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — SELL (started 2024-11-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:45:00 | 1610.90 | 1620.86 | 0.00 | ORB-short ORB[1612.01,1631.46] vol=1.5x ATR=5.44 |
| Stop hit — per-position SL triggered | 2024-11-13 09:50:00 | 1616.34 | 1620.33 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-11-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-14 09:30:00 | 1629.60 | 1623.54 | 0.00 | ORB-long ORB[1612.00,1629.00] vol=1.7x ATR=4.71 |
| Stop hit — per-position SL triggered | 2024-11-14 09:35:00 | 1624.89 | 1623.65 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-11-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 09:35:00 | 1653.71 | 1650.37 | 0.00 | ORB-long ORB[1643.05,1651.98] vol=2.0x ATR=4.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-22 10:40:00 | 1659.90 | 1653.85 | 0.00 | T1 1.5R @ 1659.90 |
| Target hit | 2024-11-22 11:45:00 | 1655.69 | 1655.94 | 0.00 | Trail-exit close<VWAP |

### Cycle 47 — BUY (started 2024-12-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:35:00 | 1753.42 | 1745.07 | 0.00 | ORB-long ORB[1735.64,1747.55] vol=2.4x ATR=4.00 |
| Stop hit — per-position SL triggered | 2024-12-04 09:40:00 | 1749.42 | 1746.06 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-12-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 11:00:00 | 1776.75 | 1765.05 | 0.00 | ORB-long ORB[1752.60,1767.11] vol=2.0x ATR=4.22 |
| Stop hit — per-position SL triggered | 2024-12-06 11:10:00 | 1772.53 | 1766.55 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-12-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 09:55:00 | 1785.84 | 1780.25 | 0.00 | ORB-long ORB[1774.01,1785.20] vol=2.1x ATR=3.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 11:30:00 | 1791.54 | 1783.43 | 0.00 | T1 1.5R @ 1791.54 |
| Target hit | 2024-12-10 15:20:00 | 1800.00 | 1794.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — BUY (started 2024-12-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 09:30:00 | 1831.68 | 1823.47 | 0.00 | ORB-long ORB[1806.09,1828.00] vol=3.5x ATR=4.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 09:45:00 | 1838.41 | 1828.25 | 0.00 | T1 1.5R @ 1838.41 |
| Target hit | 2024-12-12 11:45:00 | 1845.69 | 1847.60 | 0.00 | Trail-exit close<VWAP |

### Cycle 51 — SELL (started 2024-12-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 10:05:00 | 1851.46 | 1856.59 | 0.00 | ORB-short ORB[1853.20,1870.99] vol=2.0x ATR=4.82 |
| Stop hit — per-position SL triggered | 2024-12-16 10:15:00 | 1856.28 | 1856.35 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-12-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 09:30:00 | 1875.55 | 1867.40 | 0.00 | ORB-long ORB[1856.00,1869.59] vol=3.1x ATR=3.84 |
| Stop hit — per-position SL triggered | 2024-12-17 09:35:00 | 1871.71 | 1868.75 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 11:15:00 | 1892.54 | 1886.77 | 0.00 | ORB-long ORB[1874.00,1891.48] vol=2.1x ATR=4.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 11:50:00 | 1898.91 | 1888.63 | 0.00 | T1 1.5R @ 1898.91 |
| Stop hit — per-position SL triggered | 2024-12-18 12:35:00 | 1892.54 | 1891.07 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-12-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 10:55:00 | 1903.00 | 1890.46 | 0.00 | ORB-long ORB[1877.50,1898.77] vol=1.5x ATR=5.22 |
| Stop hit — per-position SL triggered | 2024-12-30 11:10:00 | 1897.78 | 1892.13 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:30:00 | 1939.44 | 1929.84 | 0.00 | ORB-long ORB[1920.00,1934.86] vol=2.4x ATR=5.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 09:50:00 | 1948.34 | 1936.67 | 0.00 | T1 1.5R @ 1948.34 |
| Stop hit — per-position SL triggered | 2025-01-02 10:05:00 | 1939.44 | 1938.09 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-01-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 09:30:00 | 1940.17 | 1935.90 | 0.00 | ORB-long ORB[1926.60,1939.01] vol=1.9x ATR=5.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-07 09:35:00 | 1948.80 | 1945.71 | 0.00 | T1 1.5R @ 1948.80 |
| Stop hit — per-position SL triggered | 2025-01-07 09:45:00 | 1940.17 | 1945.59 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-01-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-17 09:40:00 | 1720.74 | 1728.91 | 0.00 | ORB-short ORB[1722.38,1740.00] vol=1.5x ATR=5.71 |
| Stop hit — per-position SL triggered | 2025-01-17 09:45:00 | 1726.45 | 1728.41 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-01-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 09:40:00 | 1720.00 | 1728.50 | 0.00 | ORB-short ORB[1730.60,1746.54] vol=1.8x ATR=6.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 10:00:00 | 1709.77 | 1722.81 | 0.00 | T1 1.5R @ 1709.77 |
| Target hit | 2025-01-21 15:20:00 | 1656.42 | 1677.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — SELL (started 2025-01-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 09:45:00 | 1804.68 | 1816.32 | 0.00 | ORB-short ORB[1810.75,1825.75] vol=1.5x ATR=8.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:00:00 | 1792.27 | 1813.55 | 0.00 | T1 1.5R @ 1792.27 |
| Stop hit — per-position SL triggered | 2025-01-27 11:10:00 | 1804.68 | 1803.00 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-02-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 09:30:00 | 1711.99 | 1702.67 | 0.00 | ORB-long ORB[1689.95,1710.99] vol=1.6x ATR=5.35 |
| Stop hit — per-position SL triggered | 2025-02-07 09:40:00 | 1706.64 | 1704.42 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-10 11:15:00 | 1517.53 | 1525.15 | 0.00 | ORB-short ORB[1521.20,1541.42] vol=4.6x ATR=4.66 |
| Stop hit — per-position SL triggered | 2025-03-10 11:25:00 | 1522.19 | 1525.00 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-03-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 09:30:00 | 1468.04 | 1459.69 | 0.00 | ORB-long ORB[1449.60,1465.00] vol=1.7x ATR=4.18 |
| Stop hit — per-position SL triggered | 2025-03-17 09:35:00 | 1463.86 | 1460.10 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-04-02 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-02 10:10:00 | 1538.20 | 1552.41 | 0.00 | ORB-short ORB[1550.43,1566.88] vol=1.6x ATR=6.80 |
| Stop hit — per-position SL triggered | 2025-04-02 11:35:00 | 1545.00 | 1547.30 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-04-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 09:35:00 | 1399.40 | 1388.32 | 0.00 | ORB-long ORB[1374.70,1393.20] vol=1.7x ATR=6.00 |
| Stop hit — per-position SL triggered | 2025-04-22 09:40:00 | 1393.40 | 1388.88 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 10:15:00 | 1500.50 | 1491.34 | 0.00 | ORB-long ORB[1480.00,1499.20] vol=2.1x ATR=6.66 |
| Stop hit — per-position SL triggered | 2025-05-05 10:35:00 | 1493.84 | 1491.77 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-05-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 09:30:00 | 1542.20 | 1533.99 | 0.00 | ORB-long ORB[1524.00,1534.80] vol=2.1x ATR=5.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 09:50:00 | 1550.50 | 1539.22 | 0.00 | T1 1.5R @ 1550.50 |
| Target hit | 2025-05-08 13:10:00 | 1561.40 | 1562.13 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-15 11:00:00 | 920.77 | 2024-05-15 12:10:00 | 924.82 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-05-15 11:00:00 | 920.77 | 2024-05-15 12:45:00 | 920.77 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-17 11:15:00 | 943.02 | 2024-05-17 11:30:00 | 939.90 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-05-21 09:40:00 | 954.81 | 2024-05-21 09:50:00 | 951.36 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-05-23 09:35:00 | 995.88 | 2024-05-23 09:45:00 | 1001.49 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-05-23 09:35:00 | 995.88 | 2024-05-23 13:15:00 | 1016.40 | TARGET_HIT | 0.50 | 2.06% |
| BUY | retest1 | 2024-05-27 09:30:00 | 1038.46 | 2024-05-27 09:35:00 | 1042.97 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-05-27 09:30:00 | 1038.46 | 2024-05-27 09:40:00 | 1038.46 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-30 09:30:00 | 1011.51 | 2024-05-30 09:50:00 | 1007.38 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-05-30 09:30:00 | 1011.51 | 2024-05-30 15:20:00 | 997.30 | TARGET_HIT | 0.50 | 1.40% |
| SELL | retest1 | 2024-05-31 11:10:00 | 992.70 | 2024-05-31 11:35:00 | 995.44 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-06-18 10:15:00 | 1032.84 | 2024-06-18 10:30:00 | 1035.42 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-06-24 10:45:00 | 1084.85 | 2024-06-24 10:55:00 | 1080.81 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-06-25 10:45:00 | 1076.21 | 2024-06-25 11:10:00 | 1073.43 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-07-01 09:45:00 | 1107.28 | 2024-07-01 10:55:00 | 1114.85 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2024-07-01 09:45:00 | 1107.28 | 2024-07-01 12:15:00 | 1107.28 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-02 09:50:00 | 1118.24 | 2024-07-02 10:10:00 | 1124.16 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-07-02 09:50:00 | 1118.24 | 2024-07-02 12:35:00 | 1126.55 | TARGET_HIT | 0.50 | 0.74% |
| BUY | retest1 | 2024-07-08 09:30:00 | 1193.00 | 2024-07-08 09:35:00 | 1188.91 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-07-10 09:45:00 | 1141.69 | 2024-07-10 10:05:00 | 1136.55 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-07-10 09:45:00 | 1141.69 | 2024-07-10 11:40:00 | 1141.69 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-29 09:45:00 | 1263.42 | 2024-07-29 10:45:00 | 1267.42 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-07-30 10:45:00 | 1256.42 | 2024-07-30 10:55:00 | 1259.30 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-07-31 09:40:00 | 1268.19 | 2024-07-31 10:25:00 | 1264.37 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-08-02 10:50:00 | 1229.57 | 2024-08-02 11:40:00 | 1233.41 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-08-07 10:25:00 | 1196.01 | 2024-08-07 10:50:00 | 1201.75 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-08-07 10:25:00 | 1196.01 | 2024-08-07 11:10:00 | 1196.01 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-08 09:35:00 | 1189.79 | 2024-08-08 09:55:00 | 1193.01 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-08-13 09:35:00 | 1178.79 | 2024-08-13 09:40:00 | 1183.42 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-08-13 09:35:00 | 1178.79 | 2024-08-13 11:30:00 | 1180.39 | TARGET_HIT | 0.50 | 0.14% |
| SELL | retest1 | 2024-08-14 09:30:00 | 1161.19 | 2024-08-14 09:45:00 | 1164.69 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-08-19 09:35:00 | 1228.00 | 2024-08-19 09:45:00 | 1224.30 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-08-28 09:50:00 | 1230.60 | 2024-08-28 10:10:00 | 1236.26 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-08-28 09:50:00 | 1230.60 | 2024-08-28 15:15:00 | 1249.00 | TARGET_HIT | 0.50 | 1.50% |
| BUY | retest1 | 2024-08-29 11:05:00 | 1268.18 | 2024-08-29 11:20:00 | 1263.95 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-09-02 09:35:00 | 1280.81 | 2024-09-02 09:50:00 | 1276.60 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-09-05 10:20:00 | 1276.19 | 2024-09-05 10:35:00 | 1273.08 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-09-11 11:00:00 | 1370.06 | 2024-09-11 11:10:00 | 1366.63 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-09-13 09:35:00 | 1368.75 | 2024-09-13 09:55:00 | 1372.60 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-09-16 09:45:00 | 1392.22 | 2024-09-16 10:55:00 | 1397.49 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-09-18 09:30:00 | 1375.47 | 2024-09-18 09:40:00 | 1379.71 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-09-24 11:00:00 | 1379.50 | 2024-09-24 11:05:00 | 1384.89 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-09-24 11:00:00 | 1379.50 | 2024-09-24 11:25:00 | 1379.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-25 09:45:00 | 1411.88 | 2024-09-25 10:05:00 | 1407.73 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-09-30 10:05:00 | 1398.51 | 2024-09-30 10:20:00 | 1392.77 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-10-01 10:05:00 | 1416.00 | 2024-10-01 10:20:00 | 1411.26 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-10-07 09:35:00 | 1452.62 | 2024-10-07 09:45:00 | 1446.56 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-10-15 09:35:00 | 1513.44 | 2024-10-15 09:40:00 | 1507.70 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-10-28 11:15:00 | 1519.85 | 2024-10-28 11:20:00 | 1524.81 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-11-04 09:30:00 | 1523.74 | 2024-11-04 09:40:00 | 1516.71 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-11-06 09:40:00 | 1533.76 | 2024-11-06 09:50:00 | 1543.24 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-11-06 09:40:00 | 1533.76 | 2024-11-06 15:20:00 | 1569.87 | TARGET_HIT | 0.50 | 2.35% |
| SELL | retest1 | 2024-11-07 10:05:00 | 1556.39 | 2024-11-07 11:30:00 | 1562.60 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-11-08 09:50:00 | 1601.97 | 2024-11-08 10:10:00 | 1596.25 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-11-11 10:00:00 | 1605.30 | 2024-11-11 12:25:00 | 1613.60 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-11-11 10:00:00 | 1605.30 | 2024-11-11 15:20:00 | 1612.23 | TARGET_HIT | 0.50 | 0.43% |
| SELL | retest1 | 2024-11-13 09:45:00 | 1610.90 | 2024-11-13 09:50:00 | 1616.34 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-11-14 09:30:00 | 1629.60 | 2024-11-14 09:35:00 | 1624.89 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-11-22 09:35:00 | 1653.71 | 2024-11-22 10:40:00 | 1659.90 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-11-22 09:35:00 | 1653.71 | 2024-11-22 11:45:00 | 1655.69 | TARGET_HIT | 0.50 | 0.12% |
| BUY | retest1 | 2024-12-04 09:35:00 | 1753.42 | 2024-12-04 09:40:00 | 1749.42 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-12-06 11:00:00 | 1776.75 | 2024-12-06 11:10:00 | 1772.53 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-12-10 09:55:00 | 1785.84 | 2024-12-10 11:30:00 | 1791.54 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-12-10 09:55:00 | 1785.84 | 2024-12-10 15:20:00 | 1800.00 | TARGET_HIT | 0.50 | 0.79% |
| BUY | retest1 | 2024-12-12 09:30:00 | 1831.68 | 2024-12-12 09:45:00 | 1838.41 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-12-12 09:30:00 | 1831.68 | 2024-12-12 11:45:00 | 1845.69 | TARGET_HIT | 0.50 | 0.76% |
| SELL | retest1 | 2024-12-16 10:05:00 | 1851.46 | 2024-12-16 10:15:00 | 1856.28 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-12-17 09:30:00 | 1875.55 | 2024-12-17 09:35:00 | 1871.71 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-12-18 11:15:00 | 1892.54 | 2024-12-18 11:50:00 | 1898.91 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-12-18 11:15:00 | 1892.54 | 2024-12-18 12:35:00 | 1892.54 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-30 10:55:00 | 1903.00 | 2024-12-30 11:10:00 | 1897.78 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-01-02 09:30:00 | 1939.44 | 2025-01-02 09:50:00 | 1948.34 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-01-02 09:30:00 | 1939.44 | 2025-01-02 10:05:00 | 1939.44 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-07 09:30:00 | 1940.17 | 2025-01-07 09:35:00 | 1948.80 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-01-07 09:30:00 | 1940.17 | 2025-01-07 09:45:00 | 1940.17 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-17 09:40:00 | 1720.74 | 2025-01-17 09:45:00 | 1726.45 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-01-21 09:40:00 | 1720.00 | 2025-01-21 10:00:00 | 1709.77 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2025-01-21 09:40:00 | 1720.00 | 2025-01-21 15:20:00 | 1656.42 | TARGET_HIT | 0.50 | 3.70% |
| SELL | retest1 | 2025-01-27 09:45:00 | 1804.68 | 2025-01-27 10:00:00 | 1792.27 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2025-01-27 09:45:00 | 1804.68 | 2025-01-27 11:10:00 | 1804.68 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-07 09:30:00 | 1711.99 | 2025-02-07 09:40:00 | 1706.64 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-03-10 11:15:00 | 1517.53 | 2025-03-10 11:25:00 | 1522.19 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-03-17 09:30:00 | 1468.04 | 2025-03-17 09:35:00 | 1463.86 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-04-02 10:10:00 | 1538.20 | 2025-04-02 11:35:00 | 1545.00 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-04-22 09:35:00 | 1399.40 | 2025-04-22 09:40:00 | 1393.40 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-05-05 10:15:00 | 1500.50 | 2025-05-05 10:35:00 | 1493.84 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-05-08 09:30:00 | 1542.20 | 2025-05-08 09:50:00 | 1550.50 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-05-08 09:30:00 | 1542.20 | 2025-05-08 13:10:00 | 1561.40 | TARGET_HIT | 0.50 | 1.24% |
