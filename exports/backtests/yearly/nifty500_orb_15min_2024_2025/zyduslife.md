# Zydus Lifesciences Ltd. (ZYDUSLIFE)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 939.00
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
| PARTIAL | 26 |
| TARGET_HIT | 13 |
| STOP_HIT | 73 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 112 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 39 / 73
- **Target hits / Stop hits / Partials:** 13 / 73 / 26
- **Avg / median % per leg:** 0.06% / -0.19%
- **Sum % (uncompounded):** 7.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 58 | 20 | 34.5% | 7 | 38 | 13 | 0.09% | 5.2% |
| BUY @ 2nd Alert (retest1) | 58 | 20 | 34.5% | 7 | 38 | 13 | 0.09% | 5.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 54 | 19 | 35.2% | 6 | 35 | 13 | 0.04% | 2.0% |
| SELL @ 2nd Alert (retest1) | 54 | 19 | 35.2% | 6 | 35 | 13 | 0.04% | 2.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 112 | 39 | 34.8% | 13 | 73 | 26 | 0.06% | 7.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 10:45:00 | 1003.40 | 996.22 | 0.00 | ORB-long ORB[984.15,997.05] vol=1.9x ATR=3.80 |
| Stop hit — per-position SL triggered | 2024-05-16 11:15:00 | 999.60 | 996.99 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 09:40:00 | 1020.60 | 1017.91 | 0.00 | ORB-long ORB[1009.20,1020.00] vol=1.6x ATR=4.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 09:50:00 | 1027.19 | 1019.84 | 0.00 | T1 1.5R @ 1027.19 |
| Target hit | 2024-05-17 11:35:00 | 1026.45 | 1030.52 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — SELL (started 2024-05-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 09:45:00 | 1067.35 | 1074.14 | 0.00 | ORB-short ORB[1072.10,1086.00] vol=1.5x ATR=3.68 |
| Stop hit — per-position SL triggered | 2024-05-27 10:45:00 | 1071.03 | 1069.54 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 09:30:00 | 1016.95 | 1022.57 | 0.00 | ORB-short ORB[1023.00,1036.65] vol=4.5x ATR=4.03 |
| Stop hit — per-position SL triggered | 2024-05-31 09:35:00 | 1020.98 | 1020.89 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 10:40:00 | 1107.65 | 1100.53 | 0.00 | ORB-long ORB[1094.35,1101.25] vol=1.5x ATR=2.88 |
| Stop hit — per-position SL triggered | 2024-06-14 11:05:00 | 1104.77 | 1102.15 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 09:40:00 | 1077.80 | 1080.16 | 0.00 | ORB-short ORB[1078.00,1085.90] vol=1.5x ATR=2.87 |
| Stop hit — per-position SL triggered | 2024-06-19 09:50:00 | 1080.67 | 1080.25 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 10:35:00 | 1075.75 | 1077.43 | 0.00 | ORB-short ORB[1077.15,1086.00] vol=4.3x ATR=3.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 11:05:00 | 1070.75 | 1076.52 | 0.00 | T1 1.5R @ 1070.75 |
| Target hit | 2024-06-25 14:35:00 | 1072.95 | 1072.32 | 0.00 | Trail-exit close>VWAP |

### Cycle 8 — SELL (started 2024-06-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 10:45:00 | 1065.10 | 1071.68 | 0.00 | ORB-short ORB[1068.05,1075.00] vol=2.1x ATR=2.88 |
| Stop hit — per-position SL triggered | 2024-06-27 10:50:00 | 1067.98 | 1071.48 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 10:40:00 | 1077.40 | 1073.63 | 0.00 | ORB-long ORB[1060.50,1076.00] vol=1.5x ATR=3.51 |
| Stop hit — per-position SL triggered | 2024-06-28 10:50:00 | 1073.89 | 1074.65 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-07-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:35:00 | 1135.00 | 1151.52 | 0.00 | ORB-short ORB[1156.10,1171.00] vol=2.6x ATR=4.40 |
| Stop hit — per-position SL triggered | 2024-07-10 10:40:00 | 1139.40 | 1150.81 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 11:00:00 | 1190.60 | 1184.55 | 0.00 | ORB-long ORB[1171.70,1181.75] vol=1.9x ATR=3.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 12:05:00 | 1196.54 | 1187.70 | 0.00 | T1 1.5R @ 1196.54 |
| Stop hit — per-position SL triggered | 2024-07-15 12:55:00 | 1190.60 | 1188.75 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 11:05:00 | 1196.20 | 1191.26 | 0.00 | ORB-long ORB[1185.70,1193.50] vol=2.0x ATR=2.61 |
| Stop hit — per-position SL triggered | 2024-07-16 11:20:00 | 1193.59 | 1191.53 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:35:00 | 1174.20 | 1183.10 | 0.00 | ORB-short ORB[1183.40,1190.55] vol=1.5x ATR=3.35 |
| Stop hit — per-position SL triggered | 2024-07-18 09:40:00 | 1177.55 | 1182.15 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 10:25:00 | 1139.40 | 1145.58 | 0.00 | ORB-short ORB[1146.05,1159.00] vol=2.0x ATR=3.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 11:15:00 | 1133.92 | 1144.12 | 0.00 | T1 1.5R @ 1133.92 |
| Stop hit — per-position SL triggered | 2024-07-23 11:30:00 | 1139.40 | 1143.21 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 09:40:00 | 1162.50 | 1157.58 | 0.00 | ORB-long ORB[1147.15,1158.15] vol=1.5x ATR=4.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-24 10:10:00 | 1168.99 | 1160.58 | 0.00 | T1 1.5R @ 1168.99 |
| Stop hit — per-position SL triggered | 2024-07-24 12:55:00 | 1162.50 | 1164.71 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 11:15:00 | 1184.10 | 1178.09 | 0.00 | ORB-long ORB[1167.55,1183.25] vol=1.8x ATR=3.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 11:20:00 | 1188.94 | 1178.44 | 0.00 | T1 1.5R @ 1188.94 |
| Stop hit — per-position SL triggered | 2024-07-25 12:40:00 | 1184.10 | 1182.04 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-30 10:00:00 | 1233.00 | 1233.74 | 0.00 | ORB-short ORB[1233.90,1244.95] vol=1.5x ATR=3.92 |
| Stop hit — per-position SL triggered | 2024-07-30 10:05:00 | 1236.92 | 1233.88 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-08-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 11:10:00 | 1247.05 | 1254.51 | 0.00 | ORB-short ORB[1251.10,1265.00] vol=1.7x ATR=3.69 |
| Stop hit — per-position SL triggered | 2024-08-01 11:20:00 | 1250.74 | 1254.30 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-06 10:55:00 | 1252.30 | 1246.55 | 0.00 | ORB-long ORB[1232.55,1247.55] vol=1.8x ATR=4.33 |
| Stop hit — per-position SL triggered | 2024-08-06 11:05:00 | 1247.97 | 1246.74 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 11:10:00 | 1268.40 | 1259.78 | 0.00 | ORB-long ORB[1240.00,1254.25] vol=1.9x ATR=4.96 |
| Stop hit — per-position SL triggered | 2024-08-07 11:20:00 | 1263.44 | 1260.71 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 09:30:00 | 1288.50 | 1280.11 | 0.00 | ORB-long ORB[1270.05,1281.00] vol=1.6x ATR=5.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-08 10:35:00 | 1296.85 | 1286.08 | 0.00 | T1 1.5R @ 1296.85 |
| Stop hit — per-position SL triggered | 2024-08-08 10:45:00 | 1288.50 | 1286.75 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-22 10:00:00 | 1204.80 | 1209.04 | 0.00 | ORB-short ORB[1205.65,1220.00] vol=1.7x ATR=3.20 |
| Stop hit — per-position SL triggered | 2024-08-22 10:25:00 | 1208.00 | 1208.49 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 09:40:00 | 1201.45 | 1205.34 | 0.00 | ORB-short ORB[1202.00,1214.90] vol=1.9x ATR=3.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 09:55:00 | 1196.95 | 1204.23 | 0.00 | T1 1.5R @ 1196.95 |
| Target hit | 2024-08-23 15:20:00 | 1180.95 | 1187.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — BUY (started 2024-08-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 10:00:00 | 1147.00 | 1140.73 | 0.00 | ORB-long ORB[1136.25,1144.00] vol=1.7x ATR=3.49 |
| Stop hit — per-position SL triggered | 2024-08-30 10:10:00 | 1143.51 | 1141.98 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-09-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 09:35:00 | 1131.50 | 1125.81 | 0.00 | ORB-long ORB[1115.55,1128.00] vol=3.1x ATR=2.99 |
| Stop hit — per-position SL triggered | 2024-09-12 09:50:00 | 1128.51 | 1127.78 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-09-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 10:55:00 | 1112.15 | 1113.70 | 0.00 | ORB-short ORB[1114.10,1125.00] vol=1.8x ATR=2.03 |
| Stop hit — per-position SL triggered | 2024-09-17 11:30:00 | 1114.18 | 1113.36 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-09-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 10:55:00 | 1102.50 | 1104.61 | 0.00 | ORB-short ORB[1103.50,1111.90] vol=3.4x ATR=2.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 11:10:00 | 1099.50 | 1104.05 | 0.00 | T1 1.5R @ 1099.50 |
| Target hit | 2024-09-18 15:20:00 | 1080.35 | 1081.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — SELL (started 2024-09-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 11:10:00 | 1053.10 | 1054.86 | 0.00 | ORB-short ORB[1054.35,1063.80] vol=3.8x ATR=2.02 |
| Stop hit — per-position SL triggered | 2024-09-24 12:05:00 | 1055.12 | 1054.52 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-09-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 10:40:00 | 1043.50 | 1045.72 | 0.00 | ORB-short ORB[1050.15,1056.75] vol=2.7x ATR=2.74 |
| Stop hit — per-position SL triggered | 2024-09-25 10:50:00 | 1046.24 | 1045.35 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-09-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 10:55:00 | 1060.25 | 1060.84 | 0.00 | ORB-short ORB[1060.40,1066.15] vol=3.2x ATR=2.55 |
| Stop hit — per-position SL triggered | 2024-09-26 11:05:00 | 1062.80 | 1060.90 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-09-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 11:05:00 | 1056.30 | 1052.60 | 0.00 | ORB-long ORB[1042.65,1056.00] vol=1.8x ATR=2.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 11:40:00 | 1060.71 | 1053.42 | 0.00 | T1 1.5R @ 1060.71 |
| Target hit | 2024-09-27 15:20:00 | 1076.30 | 1062.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — SELL (started 2024-10-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:50:00 | 1050.45 | 1057.53 | 0.00 | ORB-short ORB[1054.95,1062.65] vol=1.7x ATR=3.67 |
| Stop hit — per-position SL triggered | 2024-10-07 11:10:00 | 1054.12 | 1056.54 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-10-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 09:35:00 | 1061.90 | 1057.92 | 0.00 | ORB-long ORB[1052.00,1059.70] vol=1.7x ATR=3.29 |
| Stop hit — per-position SL triggered | 2024-10-09 09:40:00 | 1058.61 | 1058.15 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-10-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 10:55:00 | 1057.10 | 1061.95 | 0.00 | ORB-short ORB[1060.45,1070.00] vol=1.8x ATR=2.48 |
| Stop hit — per-position SL triggered | 2024-10-14 11:30:00 | 1059.58 | 1060.90 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-10-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:40:00 | 994.25 | 1001.52 | 0.00 | ORB-short ORB[1002.35,1013.05] vol=3.1x ATR=3.38 |
| Stop hit — per-position SL triggered | 2024-10-22 10:55:00 | 997.63 | 999.94 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-10-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-23 10:45:00 | 1003.95 | 996.76 | 0.00 | ORB-long ORB[986.90,997.45] vol=3.6x ATR=4.05 |
| Stop hit — per-position SL triggered | 2024-10-23 10:50:00 | 999.90 | 997.01 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-11-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-04 10:00:00 | 1009.55 | 1003.67 | 0.00 | ORB-long ORB[994.65,1008.00] vol=1.9x ATR=4.29 |
| Stop hit — per-position SL triggered | 2024-11-04 10:05:00 | 1005.26 | 1003.96 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-11-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-05 10:10:00 | 978.05 | 989.27 | 0.00 | ORB-short ORB[993.15,1005.35] vol=1.7x ATR=3.83 |
| Stop hit — per-position SL triggered | 2024-11-05 10:15:00 | 981.88 | 988.07 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-11-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 09:45:00 | 984.25 | 990.53 | 0.00 | ORB-short ORB[990.10,1000.95] vol=2.0x ATR=2.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-07 09:55:00 | 980.33 | 988.51 | 0.00 | T1 1.5R @ 980.33 |
| Stop hit — per-position SL triggered | 2024-11-07 10:25:00 | 984.25 | 985.73 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-11-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-25 09:45:00 | 947.25 | 950.31 | 0.00 | ORB-short ORB[948.05,961.80] vol=1.9x ATR=2.82 |
| Stop hit — per-position SL triggered | 2024-11-25 10:15:00 | 950.07 | 949.37 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-12-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 11:10:00 | 969.05 | 974.82 | 0.00 | ORB-short ORB[975.00,985.95] vol=2.0x ATR=2.38 |
| Stop hit — per-position SL triggered | 2024-12-04 11:30:00 | 971.43 | 973.98 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 968.20 | 972.98 | 0.00 | ORB-short ORB[970.85,978.10] vol=1.6x ATR=2.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-05 11:15:00 | 964.71 | 972.25 | 0.00 | T1 1.5R @ 964.71 |
| Stop hit — per-position SL triggered | 2024-12-05 12:05:00 | 968.20 | 970.58 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-12-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 10:45:00 | 992.65 | 986.20 | 0.00 | ORB-long ORB[975.00,984.70] vol=1.9x ATR=2.16 |
| Stop hit — per-position SL triggered | 2024-12-11 11:00:00 | 990.49 | 987.09 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-12-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 11:00:00 | 976.80 | 980.79 | 0.00 | ORB-short ORB[978.85,991.00] vol=1.8x ATR=1.83 |
| Stop hit — per-position SL triggered | 2024-12-12 11:10:00 | 978.63 | 980.66 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-12-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:20:00 | 959.25 | 964.81 | 0.00 | ORB-short ORB[968.70,981.45] vol=3.8x ATR=2.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 10:45:00 | 955.56 | 963.53 | 0.00 | T1 1.5R @ 955.56 |
| Stop hit — per-position SL triggered | 2024-12-13 11:05:00 | 959.25 | 963.07 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-12-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 10:45:00 | 978.60 | 981.68 | 0.00 | ORB-short ORB[987.00,991.10] vol=2.0x ATR=2.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 11:20:00 | 975.60 | 979.98 | 0.00 | T1 1.5R @ 975.60 |
| Stop hit — per-position SL triggered | 2024-12-17 11:35:00 | 978.60 | 979.58 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-12-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 11:10:00 | 980.65 | 979.17 | 0.00 | ORB-long ORB[968.30,978.45] vol=3.3x ATR=2.67 |
| Stop hit — per-position SL triggered | 2024-12-18 11:40:00 | 977.98 | 979.26 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-12-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 11:00:00 | 969.95 | 967.86 | 0.00 | ORB-long ORB[958.00,964.35] vol=2.5x ATR=2.36 |
| Stop hit — per-position SL triggered | 2024-12-27 11:05:00 | 967.59 | 967.99 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-12-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-30 10:25:00 | 969.85 | 971.13 | 0.00 | ORB-short ORB[970.55,979.60] vol=1.7x ATR=2.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 11:15:00 | 965.59 | 970.13 | 0.00 | T1 1.5R @ 965.59 |
| Target hit | 2024-12-30 15:20:00 | 961.55 | 958.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — BUY (started 2025-01-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 11:05:00 | 974.05 | 969.84 | 0.00 | ORB-long ORB[965.30,973.35] vol=2.0x ATR=1.85 |
| Stop hit — per-position SL triggered | 2025-01-01 11:55:00 | 972.20 | 970.77 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-01-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 09:40:00 | 961.00 | 967.25 | 0.00 | ORB-short ORB[967.00,975.80] vol=1.8x ATR=2.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 10:35:00 | 956.61 | 962.99 | 0.00 | T1 1.5R @ 956.61 |
| Target hit | 2025-01-06 11:25:00 | 960.60 | 959.96 | 0.00 | Trail-exit close>VWAP |

### Cycle 52 — BUY (started 2025-01-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 09:30:00 | 978.20 | 975.72 | 0.00 | ORB-long ORB[969.90,978.00] vol=3.0x ATR=2.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-07 09:40:00 | 982.35 | 978.69 | 0.00 | T1 1.5R @ 982.35 |
| Target hit | 2025-01-07 12:30:00 | 999.60 | 1000.51 | 0.00 | Trail-exit close<VWAP |

### Cycle 53 — SELL (started 2025-01-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 10:55:00 | 989.70 | 996.67 | 0.00 | ORB-short ORB[999.40,1012.90] vol=2.7x ATR=3.42 |
| Stop hit — per-position SL triggered | 2025-01-08 11:25:00 | 993.12 | 995.65 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-01-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-10 10:50:00 | 1015.55 | 1009.33 | 0.00 | ORB-long ORB[1002.75,1014.95] vol=1.6x ATR=4.04 |
| Stop hit — per-position SL triggered | 2025-01-10 11:40:00 | 1011.51 | 1011.84 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-01-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 11:10:00 | 976.85 | 985.82 | 0.00 | ORB-short ORB[985.30,997.90] vol=2.0x ATR=3.22 |
| Stop hit — per-position SL triggered | 2025-01-13 11:50:00 | 980.07 | 983.08 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-01-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-16 10:30:00 | 983.85 | 986.05 | 0.00 | ORB-short ORB[985.00,994.80] vol=1.6x ATR=2.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-16 10:40:00 | 980.63 | 985.74 | 0.00 | T1 1.5R @ 980.63 |
| Stop hit — per-position SL triggered | 2025-01-16 10:55:00 | 983.85 | 984.97 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 11:15:00 | 993.45 | 992.27 | 0.00 | ORB-long ORB[980.95,992.00] vol=2.5x ATR=2.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 11:55:00 | 997.26 | 992.55 | 0.00 | T1 1.5R @ 997.26 |
| Stop hit — per-position SL triggered | 2025-01-17 13:00:00 | 993.45 | 994.44 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-01-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-22 09:50:00 | 974.60 | 980.76 | 0.00 | ORB-short ORB[977.40,989.25] vol=1.7x ATR=2.93 |
| Stop hit — per-position SL triggered | 2025-01-22 09:55:00 | 977.53 | 980.53 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-01-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 09:50:00 | 938.00 | 943.38 | 0.00 | ORB-short ORB[944.50,951.70] vol=1.8x ATR=3.41 |
| Stop hit — per-position SL triggered | 2025-01-27 09:55:00 | 941.41 | 942.87 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-01-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 11:00:00 | 937.60 | 926.79 | 0.00 | ORB-long ORB[910.25,923.75] vol=2.1x ATR=3.00 |
| Stop hit — per-position SL triggered | 2025-01-29 11:20:00 | 934.60 | 928.70 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:30:00 | 948.75 | 943.75 | 0.00 | ORB-long ORB[930.50,941.80] vol=6.6x ATR=3.14 |
| Stop hit — per-position SL triggered | 2025-01-30 10:30:00 | 945.61 | 945.92 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-01-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 09:30:00 | 963.85 | 959.16 | 0.00 | ORB-long ORB[953.80,962.85] vol=1.5x ATR=3.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-31 09:40:00 | 969.09 | 963.90 | 0.00 | T1 1.5R @ 969.09 |
| Target hit | 2025-01-31 15:20:00 | 970.50 | 968.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — BUY (started 2025-02-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 09:35:00 | 979.30 | 973.22 | 0.00 | ORB-long ORB[966.00,977.00] vol=3.0x ATR=2.97 |
| Stop hit — per-position SL triggered | 2025-02-05 09:45:00 | 976.33 | 973.93 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 10:15:00 | 889.25 | 891.30 | 0.00 | ORB-short ORB[893.25,906.40] vol=12.9x ATR=2.98 |
| Stop hit — per-position SL triggered | 2025-02-21 11:30:00 | 892.23 | 891.37 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-03-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-04 10:50:00 | 871.45 | 867.66 | 0.00 | ORB-long ORB[859.85,869.00] vol=1.5x ATR=2.89 |
| Stop hit — per-position SL triggered | 2025-03-04 11:00:00 | 868.56 | 867.91 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-03-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 09:40:00 | 881.00 | 877.70 | 0.00 | ORB-long ORB[871.20,879.00] vol=2.9x ATR=2.53 |
| Stop hit — per-position SL triggered | 2025-03-05 10:15:00 | 878.47 | 879.02 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-03-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 10:35:00 | 912.45 | 909.95 | 0.00 | ORB-long ORB[901.05,909.15] vol=1.9x ATR=2.93 |
| Stop hit — per-position SL triggered | 2025-03-07 11:05:00 | 909.52 | 910.68 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-03-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 10:05:00 | 908.90 | 902.47 | 0.00 | ORB-long ORB[894.00,905.00] vol=2.5x ATR=2.87 |
| Stop hit — per-position SL triggered | 2025-03-10 10:25:00 | 906.03 | 902.99 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-03-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 10:10:00 | 899.60 | 892.75 | 0.00 | ORB-long ORB[882.85,894.40] vol=1.6x ATR=3.87 |
| Target hit | 2025-03-11 15:20:00 | 900.05 | 898.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 70 — BUY (started 2025-03-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 09:30:00 | 900.45 | 895.35 | 0.00 | ORB-long ORB[887.15,899.50] vol=2.4x ATR=2.52 |
| Stop hit — per-position SL triggered | 2025-03-17 09:35:00 | 897.93 | 895.63 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-03-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:00:00 | 903.35 | 899.69 | 0.00 | ORB-long ORB[893.80,900.00] vol=1.7x ATR=2.28 |
| Stop hit — per-position SL triggered | 2025-03-18 11:55:00 | 901.07 | 901.79 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-03-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 10:55:00 | 919.40 | 915.19 | 0.00 | ORB-long ORB[908.00,917.90] vol=2.0x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-20 11:35:00 | 922.57 | 916.58 | 0.00 | T1 1.5R @ 922.57 |
| Stop hit — per-position SL triggered | 2025-03-20 13:15:00 | 919.40 | 917.77 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2025-03-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:45:00 | 924.80 | 921.48 | 0.00 | ORB-long ORB[914.05,922.00] vol=1.8x ATR=1.89 |
| Stop hit — per-position SL triggered | 2025-03-21 09:50:00 | 922.91 | 921.54 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2025-03-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 09:40:00 | 906.15 | 909.88 | 0.00 | ORB-short ORB[907.80,915.70] vol=1.6x ATR=2.51 |
| Stop hit — per-position SL triggered | 2025-03-26 10:05:00 | 908.66 | 908.76 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2025-03-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-27 10:45:00 | 890.30 | 895.87 | 0.00 | ORB-short ORB[893.60,906.50] vol=1.6x ATR=2.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-27 11:05:00 | 886.08 | 893.96 | 0.00 | T1 1.5R @ 886.08 |
| Target hit | 2025-03-27 12:40:00 | 888.10 | 887.90 | 0.00 | Trail-exit close>VWAP |

### Cycle 76 — BUY (started 2025-04-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 10:45:00 | 878.50 | 871.98 | 0.00 | ORB-long ORB[862.80,874.50] vol=4.0x ATR=2.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 11:10:00 | 882.46 | 873.04 | 0.00 | T1 1.5R @ 882.46 |
| Target hit | 2025-04-02 15:20:00 | 890.50 | 880.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 77 — BUY (started 2025-04-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 10:30:00 | 884.70 | 880.84 | 0.00 | ORB-long ORB[873.95,883.15] vol=3.9x ATR=2.80 |
| Stop hit — per-position SL triggered | 2025-04-16 10:55:00 | 881.90 | 881.89 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2025-04-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 09:45:00 | 848.25 | 842.27 | 0.00 | ORB-long ORB[837.90,845.45] vol=1.5x ATR=2.19 |
| Stop hit — per-position SL triggered | 2025-04-22 10:05:00 | 846.06 | 843.25 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2025-04-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 10:25:00 | 864.40 | 858.95 | 0.00 | ORB-long ORB[852.70,862.40] vol=2.0x ATR=2.74 |
| Stop hit — per-position SL triggered | 2025-04-23 10:30:00 | 861.66 | 859.34 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2025-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 10:15:00 | 881.00 | 873.84 | 0.00 | ORB-long ORB[861.50,873.70] vol=2.9x ATR=2.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-24 10:45:00 | 884.81 | 876.42 | 0.00 | T1 1.5R @ 884.81 |
| Stop hit — per-position SL triggered | 2025-04-24 11:30:00 | 881.00 | 878.52 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2025-04-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 10:55:00 | 863.05 | 865.90 | 0.00 | ORB-short ORB[875.60,882.35] vol=5.3x ATR=3.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 11:50:00 | 858.54 | 864.27 | 0.00 | T1 1.5R @ 858.54 |
| Stop hit — per-position SL triggered | 2025-04-25 12:00:00 | 863.05 | 864.17 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2025-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-28 09:30:00 | 871.20 | 868.63 | 0.00 | ORB-long ORB[861.25,870.90] vol=1.7x ATR=2.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-28 09:50:00 | 875.12 | 869.88 | 0.00 | T1 1.5R @ 875.12 |
| Target hit | 2025-04-28 15:20:00 | 886.25 | 881.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 83 — SELL (started 2025-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 10:15:00 | 882.45 | 884.67 | 0.00 | ORB-short ORB[887.00,893.90] vol=5.9x ATR=2.94 |
| Stop hit — per-position SL triggered | 2025-04-29 10:25:00 | 885.39 | 884.69 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2025-04-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 10:55:00 | 900.45 | 897.21 | 0.00 | ORB-long ORB[887.20,899.90] vol=2.1x ATR=2.56 |
| Stop hit — per-position SL triggered | 2025-04-30 11:05:00 | 897.89 | 897.31 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2025-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-02 11:15:00 | 876.45 | 886.99 | 0.00 | ORB-short ORB[882.75,893.25] vol=2.0x ATR=2.63 |
| Stop hit — per-position SL triggered | 2025-05-02 11:25:00 | 879.08 | 886.05 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2025-05-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-09 09:40:00 | 875.45 | 870.43 | 0.00 | ORB-long ORB[859.00,871.00] vol=3.0x ATR=3.68 |
| Stop hit — per-position SL triggered | 2025-05-09 09:55:00 | 871.77 | 871.68 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-16 10:45:00 | 1003.40 | 2024-05-16 11:15:00 | 999.60 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-05-17 09:40:00 | 1020.60 | 2024-05-17 09:50:00 | 1027.19 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-05-17 09:40:00 | 1020.60 | 2024-05-17 11:35:00 | 1026.45 | TARGET_HIT | 0.50 | 0.57% |
| SELL | retest1 | 2024-05-27 09:45:00 | 1067.35 | 2024-05-27 10:45:00 | 1071.03 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-05-31 09:30:00 | 1016.95 | 2024-05-31 09:35:00 | 1020.98 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-06-14 10:40:00 | 1107.65 | 2024-06-14 11:05:00 | 1104.77 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-06-19 09:40:00 | 1077.80 | 2024-06-19 09:50:00 | 1080.67 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-06-25 10:35:00 | 1075.75 | 2024-06-25 11:05:00 | 1070.75 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-06-25 10:35:00 | 1075.75 | 2024-06-25 14:35:00 | 1072.95 | TARGET_HIT | 0.50 | 0.26% |
| SELL | retest1 | 2024-06-27 10:45:00 | 1065.10 | 2024-06-27 10:50:00 | 1067.98 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-06-28 10:40:00 | 1077.40 | 2024-06-28 10:50:00 | 1073.89 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-07-10 10:35:00 | 1135.00 | 2024-07-10 10:40:00 | 1139.40 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-07-15 11:00:00 | 1190.60 | 2024-07-15 12:05:00 | 1196.54 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-07-15 11:00:00 | 1190.60 | 2024-07-15 12:55:00 | 1190.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-16 11:05:00 | 1196.20 | 2024-07-16 11:20:00 | 1193.59 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-07-18 09:35:00 | 1174.20 | 2024-07-18 09:40:00 | 1177.55 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-07-23 10:25:00 | 1139.40 | 2024-07-23 11:15:00 | 1133.92 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-07-23 10:25:00 | 1139.40 | 2024-07-23 11:30:00 | 1139.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-24 09:40:00 | 1162.50 | 2024-07-24 10:10:00 | 1168.99 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-07-24 09:40:00 | 1162.50 | 2024-07-24 12:55:00 | 1162.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-25 11:15:00 | 1184.10 | 2024-07-25 11:20:00 | 1188.94 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-07-25 11:15:00 | 1184.10 | 2024-07-25 12:40:00 | 1184.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-30 10:00:00 | 1233.00 | 2024-07-30 10:05:00 | 1236.92 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-08-01 11:10:00 | 1247.05 | 2024-08-01 11:20:00 | 1250.74 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-08-06 10:55:00 | 1252.30 | 2024-08-06 11:05:00 | 1247.97 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-08-07 11:10:00 | 1268.40 | 2024-08-07 11:20:00 | 1263.44 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-08-08 09:30:00 | 1288.50 | 2024-08-08 10:35:00 | 1296.85 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-08-08 09:30:00 | 1288.50 | 2024-08-08 10:45:00 | 1288.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-22 10:00:00 | 1204.80 | 2024-08-22 10:25:00 | 1208.00 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-08-23 09:40:00 | 1201.45 | 2024-08-23 09:55:00 | 1196.95 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-08-23 09:40:00 | 1201.45 | 2024-08-23 15:20:00 | 1180.95 | TARGET_HIT | 0.50 | 1.71% |
| BUY | retest1 | 2024-08-30 10:00:00 | 1147.00 | 2024-08-30 10:10:00 | 1143.51 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-09-12 09:35:00 | 1131.50 | 2024-09-12 09:50:00 | 1128.51 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-09-17 10:55:00 | 1112.15 | 2024-09-17 11:30:00 | 1114.18 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-09-18 10:55:00 | 1102.50 | 2024-09-18 11:10:00 | 1099.50 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2024-09-18 10:55:00 | 1102.50 | 2024-09-18 15:20:00 | 1080.35 | TARGET_HIT | 0.50 | 2.01% |
| SELL | retest1 | 2024-09-24 11:10:00 | 1053.10 | 2024-09-24 12:05:00 | 1055.12 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-09-25 10:40:00 | 1043.50 | 2024-09-25 10:50:00 | 1046.24 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-09-26 10:55:00 | 1060.25 | 2024-09-26 11:05:00 | 1062.80 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-09-27 11:05:00 | 1056.30 | 2024-09-27 11:40:00 | 1060.71 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-09-27 11:05:00 | 1056.30 | 2024-09-27 15:20:00 | 1076.30 | TARGET_HIT | 0.50 | 1.89% |
| SELL | retest1 | 2024-10-07 10:50:00 | 1050.45 | 2024-10-07 11:10:00 | 1054.12 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-10-09 09:35:00 | 1061.90 | 2024-10-09 09:40:00 | 1058.61 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-10-14 10:55:00 | 1057.10 | 2024-10-14 11:30:00 | 1059.58 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-10-22 10:40:00 | 994.25 | 2024-10-22 10:55:00 | 997.63 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-10-23 10:45:00 | 1003.95 | 2024-10-23 10:50:00 | 999.90 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-11-04 10:00:00 | 1009.55 | 2024-11-04 10:05:00 | 1005.26 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-11-05 10:10:00 | 978.05 | 2024-11-05 10:15:00 | 981.88 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-11-07 09:45:00 | 984.25 | 2024-11-07 09:55:00 | 980.33 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-11-07 09:45:00 | 984.25 | 2024-11-07 10:25:00 | 984.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-25 09:45:00 | 947.25 | 2024-11-25 10:15:00 | 950.07 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-12-04 11:10:00 | 969.05 | 2024-12-04 11:30:00 | 971.43 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-12-05 10:55:00 | 968.20 | 2024-12-05 11:15:00 | 964.71 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-12-05 10:55:00 | 968.20 | 2024-12-05 12:05:00 | 968.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-11 10:45:00 | 992.65 | 2024-12-11 11:00:00 | 990.49 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-12 11:00:00 | 976.80 | 2024-12-12 11:10:00 | 978.63 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-12-13 10:20:00 | 959.25 | 2024-12-13 10:45:00 | 955.56 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-12-13 10:20:00 | 959.25 | 2024-12-13 11:05:00 | 959.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-17 10:45:00 | 978.60 | 2024-12-17 11:20:00 | 975.60 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-12-17 10:45:00 | 978.60 | 2024-12-17 11:35:00 | 978.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-18 11:10:00 | 980.65 | 2024-12-18 11:40:00 | 977.98 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-12-27 11:00:00 | 969.95 | 2024-12-27 11:05:00 | 967.59 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-12-30 10:25:00 | 969.85 | 2024-12-30 11:15:00 | 965.59 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-12-30 10:25:00 | 969.85 | 2024-12-30 15:20:00 | 961.55 | TARGET_HIT | 0.50 | 0.86% |
| BUY | retest1 | 2025-01-01 11:05:00 | 974.05 | 2025-01-01 11:55:00 | 972.20 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-01-06 09:40:00 | 961.00 | 2025-01-06 10:35:00 | 956.61 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-01-06 09:40:00 | 961.00 | 2025-01-06 11:25:00 | 960.60 | TARGET_HIT | 0.50 | 0.04% |
| BUY | retest1 | 2025-01-07 09:30:00 | 978.20 | 2025-01-07 09:40:00 | 982.35 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-01-07 09:30:00 | 978.20 | 2025-01-07 12:30:00 | 999.60 | TARGET_HIT | 0.50 | 2.19% |
| SELL | retest1 | 2025-01-08 10:55:00 | 989.70 | 2025-01-08 11:25:00 | 993.12 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-01-10 10:50:00 | 1015.55 | 2025-01-10 11:40:00 | 1011.51 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-01-13 11:10:00 | 976.85 | 2025-01-13 11:50:00 | 980.07 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-01-16 10:30:00 | 983.85 | 2025-01-16 10:40:00 | 980.63 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-01-16 10:30:00 | 983.85 | 2025-01-16 10:55:00 | 983.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-17 11:15:00 | 993.45 | 2025-01-17 11:55:00 | 997.26 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-01-17 11:15:00 | 993.45 | 2025-01-17 13:00:00 | 993.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-22 09:50:00 | 974.60 | 2025-01-22 09:55:00 | 977.53 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-01-27 09:50:00 | 938.00 | 2025-01-27 09:55:00 | 941.41 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-01-29 11:00:00 | 937.60 | 2025-01-29 11:20:00 | 934.60 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-01-30 09:30:00 | 948.75 | 2025-01-30 10:30:00 | 945.61 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-01-31 09:30:00 | 963.85 | 2025-01-31 09:40:00 | 969.09 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-01-31 09:30:00 | 963.85 | 2025-01-31 15:20:00 | 970.50 | TARGET_HIT | 0.50 | 0.69% |
| BUY | retest1 | 2025-02-05 09:35:00 | 979.30 | 2025-02-05 09:45:00 | 976.33 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-02-21 10:15:00 | 889.25 | 2025-02-21 11:30:00 | 892.23 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-03-04 10:50:00 | 871.45 | 2025-03-04 11:00:00 | 868.56 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-03-05 09:40:00 | 881.00 | 2025-03-05 10:15:00 | 878.47 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-03-07 10:35:00 | 912.45 | 2025-03-07 11:05:00 | 909.52 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-03-10 10:05:00 | 908.90 | 2025-03-10 10:25:00 | 906.03 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-03-11 10:10:00 | 899.60 | 2025-03-11 15:20:00 | 900.05 | TARGET_HIT | 1.00 | 0.05% |
| BUY | retest1 | 2025-03-17 09:30:00 | 900.45 | 2025-03-17 09:35:00 | 897.93 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-03-18 10:00:00 | 903.35 | 2025-03-18 11:55:00 | 901.07 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-03-20 10:55:00 | 919.40 | 2025-03-20 11:35:00 | 922.57 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-03-20 10:55:00 | 919.40 | 2025-03-20 13:15:00 | 919.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-21 09:45:00 | 924.80 | 2025-03-21 09:50:00 | 922.91 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-03-26 09:40:00 | 906.15 | 2025-03-26 10:05:00 | 908.66 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-03-27 10:45:00 | 890.30 | 2025-03-27 11:05:00 | 886.08 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-03-27 10:45:00 | 890.30 | 2025-03-27 12:40:00 | 888.10 | TARGET_HIT | 0.50 | 0.25% |
| BUY | retest1 | 2025-04-02 10:45:00 | 878.50 | 2025-04-02 11:10:00 | 882.46 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-04-02 10:45:00 | 878.50 | 2025-04-02 15:20:00 | 890.50 | TARGET_HIT | 0.50 | 1.37% |
| BUY | retest1 | 2025-04-16 10:30:00 | 884.70 | 2025-04-16 10:55:00 | 881.90 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-04-22 09:45:00 | 848.25 | 2025-04-22 10:05:00 | 846.06 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-04-23 10:25:00 | 864.40 | 2025-04-23 10:30:00 | 861.66 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-04-24 10:15:00 | 881.00 | 2025-04-24 10:45:00 | 884.81 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-04-24 10:15:00 | 881.00 | 2025-04-24 11:30:00 | 881.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-25 10:55:00 | 863.05 | 2025-04-25 11:50:00 | 858.54 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-04-25 10:55:00 | 863.05 | 2025-04-25 12:00:00 | 863.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-28 09:30:00 | 871.20 | 2025-04-28 09:50:00 | 875.12 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-04-28 09:30:00 | 871.20 | 2025-04-28 15:20:00 | 886.25 | TARGET_HIT | 0.50 | 1.73% |
| SELL | retest1 | 2025-04-29 10:15:00 | 882.45 | 2025-04-29 10:25:00 | 885.39 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-04-30 10:55:00 | 900.45 | 2025-04-30 11:05:00 | 897.89 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-05-02 11:15:00 | 876.45 | 2025-05-02 11:25:00 | 879.08 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-05-09 09:40:00 | 875.45 | 2025-05-09 09:55:00 | 871.77 | STOP_HIT | 1.00 | -0.42% |
