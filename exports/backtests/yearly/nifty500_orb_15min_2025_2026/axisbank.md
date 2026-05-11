# Axis Bank Ltd. (AXISBANK)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (16813 bars)
- **Last close:** 1270.00
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
| ENTRY1 | 78 |
| ENTRY2 | 0 |
| PARTIAL | 35 |
| TARGET_HIT | 16 |
| STOP_HIT | 62 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 113 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 51 / 62
- **Target hits / Stop hits / Partials:** 16 / 62 / 35
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 17.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 55 | 26 | 47.3% | 9 | 29 | 17 | 0.14% | 7.9% |
| BUY @ 2nd Alert (retest1) | 55 | 26 | 47.3% | 9 | 29 | 17 | 0.14% | 7.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 58 | 25 | 43.1% | 7 | 33 | 18 | 0.17% | 9.6% |
| SELL @ 2nd Alert (retest1) | 58 | 25 | 43.1% | 7 | 33 | 18 | 0.17% | 9.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 113 | 51 | 45.1% | 16 | 62 | 35 | 0.15% | 17.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-16 09:30:00 | 1203.70 | 1208.19 | 0.00 | ORB-short ORB[1204.10,1213.30] vol=1.6x ATR=3.57 |
| Stop hit — per-position SL triggered | 2025-05-16 09:40:00 | 1207.27 | 1207.81 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 11:05:00 | 1201.70 | 1200.49 | 0.00 | ORB-long ORB[1196.10,1201.20] vol=2.2x ATR=2.42 |
| Stop hit — per-position SL triggered | 2025-05-21 11:10:00 | 1199.28 | 1200.47 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 10:45:00 | 1205.90 | 1197.59 | 0.00 | ORB-long ORB[1188.10,1194.40] vol=2.6x ATR=2.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-23 10:50:00 | 1209.91 | 1199.03 | 0.00 | T1 1.5R @ 1209.91 |
| Stop hit — per-position SL triggered | 2025-05-23 11:45:00 | 1205.90 | 1202.07 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-06-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 09:30:00 | 1185.00 | 1188.83 | 0.00 | ORB-short ORB[1185.30,1198.90] vol=1.6x ATR=2.73 |
| Stop hit — per-position SL triggered | 2025-06-03 09:35:00 | 1187.73 | 1188.69 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 10:50:00 | 1175.70 | 1179.32 | 0.00 | ORB-short ORB[1177.80,1191.30] vol=1.8x ATR=2.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-04 11:15:00 | 1172.58 | 1178.51 | 0.00 | T1 1.5R @ 1172.58 |
| Stop hit — per-position SL triggered | 2025-06-04 11:40:00 | 1175.70 | 1177.89 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-06 10:10:00 | 1153.50 | 1157.17 | 0.00 | ORB-short ORB[1157.30,1163.10] vol=2.2x ATR=2.67 |
| Stop hit — per-position SL triggered | 2025-06-06 10:20:00 | 1156.17 | 1157.09 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-07-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-11 09:40:00 | 1175.60 | 1170.58 | 0.00 | ORB-long ORB[1160.60,1172.60] vol=2.0x ATR=2.58 |
| Stop hit — per-position SL triggered | 2025-07-11 09:45:00 | 1173.02 | 1170.73 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-29 10:15:00 | 1062.00 | 1067.11 | 0.00 | ORB-short ORB[1071.70,1078.50] vol=9.3x ATR=2.99 |
| Stop hit — per-position SL triggered | 2025-07-29 10:20:00 | 1064.99 | 1066.90 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-07-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 09:40:00 | 1073.50 | 1066.95 | 0.00 | ORB-long ORB[1062.00,1071.50] vol=1.6x ATR=3.33 |
| Stop hit — per-position SL triggered | 2025-07-30 09:45:00 | 1070.17 | 1067.10 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-08-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-04 09:45:00 | 1065.40 | 1063.33 | 0.00 | ORB-long ORB[1059.00,1064.40] vol=1.9x ATR=2.25 |
| Stop hit — per-position SL triggered | 2025-08-04 10:00:00 | 1063.15 | 1063.40 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-08 10:15:00 | 1063.80 | 1069.08 | 0.00 | ORB-short ORB[1068.10,1080.40] vol=1.7x ATR=2.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 11:00:00 | 1060.47 | 1067.08 | 0.00 | T1 1.5R @ 1060.47 |
| Stop hit — per-position SL triggered | 2025-08-08 12:55:00 | 1063.80 | 1063.89 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-08-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-11 10:00:00 | 1061.00 | 1057.06 | 0.00 | ORB-long ORB[1054.30,1060.00] vol=1.9x ATR=3.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 10:25:00 | 1065.65 | 1057.95 | 0.00 | T1 1.5R @ 1065.65 |
| Target hit | 2025-08-11 15:20:00 | 1071.70 | 1064.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-12 10:15:00 | 1075.10 | 1073.87 | 0.00 | ORB-long ORB[1069.70,1075.00] vol=2.0x ATR=1.69 |
| Stop hit — per-position SL triggered | 2025-08-12 10:25:00 | 1073.41 | 1073.92 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-14 11:15:00 | 1068.30 | 1068.18 | 0.00 | ORB-long ORB[1062.80,1068.00] vol=12.7x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-14 11:20:00 | 1070.07 | 1068.20 | 0.00 | T1 1.5R @ 1070.07 |
| Stop hit — per-position SL triggered | 2025-08-14 11:25:00 | 1068.30 | 1068.19 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 11:15:00 | 1089.80 | 1082.81 | 0.00 | ORB-long ORB[1075.80,1082.80] vol=2.1x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-19 11:30:00 | 1092.33 | 1084.14 | 0.00 | T1 1.5R @ 1092.33 |
| Stop hit — per-position SL triggered | 2025-08-19 13:00:00 | 1089.80 | 1086.83 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-08-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 10:35:00 | 1071.50 | 1074.05 | 0.00 | ORB-short ORB[1073.10,1078.50] vol=2.5x ATR=1.56 |
| Stop hit — per-position SL triggered | 2025-08-22 10:45:00 | 1073.06 | 1073.90 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-08-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 10:20:00 | 1061.40 | 1062.25 | 0.00 | ORB-short ORB[1061.50,1069.80] vol=1.9x ATR=1.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 10:45:00 | 1058.87 | 1061.15 | 0.00 | T1 1.5R @ 1058.87 |
| Target hit | 2025-08-26 15:20:00 | 1050.30 | 1054.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — BUY (started 2025-08-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-29 11:00:00 | 1058.20 | 1056.31 | 0.00 | ORB-long ORB[1049.60,1054.70] vol=1.7x ATR=1.70 |
| Stop hit — per-position SL triggered | 2025-08-29 12:35:00 | 1056.50 | 1056.91 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-09-08 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 10:25:00 | 1062.20 | 1058.75 | 0.00 | ORB-long ORB[1055.90,1060.90] vol=1.6x ATR=2.03 |
| Stop hit — per-position SL triggered | 2025-09-08 10:35:00 | 1060.17 | 1059.01 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-09-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-09 10:40:00 | 1050.40 | 1052.97 | 0.00 | ORB-short ORB[1053.30,1059.00] vol=3.6x ATR=1.56 |
| Stop hit — per-position SL triggered | 2025-09-09 11:00:00 | 1051.96 | 1052.24 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-09-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 11:10:00 | 1069.90 | 1062.27 | 0.00 | ORB-long ORB[1055.10,1061.00] vol=2.4x ATR=2.00 |
| Stop hit — per-position SL triggered | 2025-09-10 11:25:00 | 1067.90 | 1063.24 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-09-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 11:10:00 | 1074.20 | 1072.56 | 0.00 | ORB-long ORB[1069.00,1073.50] vol=2.4x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-11 11:25:00 | 1076.84 | 1072.83 | 0.00 | T1 1.5R @ 1076.84 |
| Stop hit — per-position SL triggered | 2025-09-11 11:35:00 | 1074.20 | 1072.97 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-09-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 09:45:00 | 1105.00 | 1098.23 | 0.00 | ORB-long ORB[1088.40,1098.70] vol=2.5x ATR=3.30 |
| Stop hit — per-position SL triggered | 2025-09-12 10:15:00 | 1101.70 | 1100.06 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-09-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 09:30:00 | 1116.20 | 1111.07 | 0.00 | ORB-long ORB[1104.00,1113.90] vol=2.3x ATR=1.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 09:35:00 | 1119.07 | 1112.82 | 0.00 | T1 1.5R @ 1119.07 |
| Stop hit — per-position SL triggered | 2025-09-16 09:45:00 | 1116.20 | 1113.84 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-09-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 11:00:00 | 1158.60 | 1161.94 | 0.00 | ORB-short ORB[1160.20,1172.80] vol=1.8x ATR=2.27 |
| Stop hit — per-position SL triggered | 2025-09-24 11:20:00 | 1160.87 | 1161.59 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-09-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-30 10:50:00 | 1129.90 | 1130.28 | 0.00 | ORB-short ORB[1130.00,1137.00] vol=6.7x ATR=2.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 11:40:00 | 1126.74 | 1130.14 | 0.00 | T1 1.5R @ 1126.74 |
| Stop hit — per-position SL triggered | 2025-09-30 12:00:00 | 1129.90 | 1129.95 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 11:15:00 | 1151.00 | 1138.31 | 0.00 | ORB-long ORB[1131.00,1141.50] vol=2.0x ATR=3.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 11:25:00 | 1156.29 | 1139.31 | 0.00 | T1 1.5R @ 1156.29 |
| Target hit | 2025-10-01 15:20:00 | 1159.90 | 1153.67 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — SELL (started 2025-10-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 11:10:00 | 1194.80 | 1197.46 | 0.00 | ORB-short ORB[1197.40,1212.00] vol=3.2x ATR=3.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 11:20:00 | 1190.08 | 1197.14 | 0.00 | T1 1.5R @ 1190.08 |
| Stop hit — per-position SL triggered | 2025-10-07 13:00:00 | 1194.80 | 1195.12 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-10-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-09 10:00:00 | 1175.70 | 1178.57 | 0.00 | ORB-short ORB[1178.00,1186.70] vol=3.1x ATR=2.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 10:15:00 | 1171.40 | 1177.29 | 0.00 | T1 1.5R @ 1171.40 |
| Target hit | 2025-10-09 11:35:00 | 1173.90 | 1173.38 | 0.00 | Trail-exit close>VWAP |

### Cycle 30 — BUY (started 2025-10-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:35:00 | 1178.50 | 1176.52 | 0.00 | ORB-long ORB[1165.20,1173.70] vol=7.8x ATR=2.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-10 09:45:00 | 1182.76 | 1177.29 | 0.00 | T1 1.5R @ 1182.76 |
| Target hit | 2025-10-10 14:15:00 | 1183.60 | 1183.74 | 0.00 | Trail-exit close<VWAP |

### Cycle 31 — BUY (started 2025-10-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 09:40:00 | 1262.60 | 1255.17 | 0.00 | ORB-long ORB[1241.30,1260.00] vol=2.3x ATR=4.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-23 10:05:00 | 1268.68 | 1258.53 | 0.00 | T1 1.5R @ 1268.68 |
| Target hit | 2025-10-23 12:45:00 | 1265.60 | 1266.19 | 0.00 | Trail-exit close<VWAP |

### Cycle 32 — SELL (started 2025-10-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 11:05:00 | 1245.70 | 1249.50 | 0.00 | ORB-short ORB[1247.00,1254.80] vol=1.8x ATR=2.54 |
| Stop hit — per-position SL triggered | 2025-10-24 11:10:00 | 1248.24 | 1249.29 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-10-28 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 10:10:00 | 1245.40 | 1251.03 | 0.00 | ORB-short ORB[1246.60,1258.00] vol=1.5x ATR=3.07 |
| Stop hit — per-position SL triggered | 2025-10-28 10:45:00 | 1248.47 | 1250.05 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-29 10:15:00 | 1240.30 | 1242.05 | 0.00 | ORB-short ORB[1242.70,1249.40] vol=5.7x ATR=2.47 |
| Stop hit — per-position SL triggered | 2025-10-29 10:25:00 | 1242.77 | 1242.10 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-11-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 09:35:00 | 1231.00 | 1227.05 | 0.00 | ORB-long ORB[1223.10,1230.00] vol=3.5x ATR=2.79 |
| Stop hit — per-position SL triggered | 2025-11-03 09:40:00 | 1228.21 | 1227.64 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-11-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 09:35:00 | 1231.20 | 1226.15 | 0.00 | ORB-long ORB[1220.00,1226.70] vol=2.8x ATR=2.63 |
| Stop hit — per-position SL triggered | 2025-11-10 09:40:00 | 1228.57 | 1226.43 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-11-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 10:50:00 | 1228.50 | 1221.66 | 0.00 | ORB-long ORB[1213.20,1220.90] vol=2.4x ATR=2.18 |
| Stop hit — per-position SL triggered | 2025-11-13 11:15:00 | 1226.32 | 1222.79 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-11-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 10:50:00 | 1233.10 | 1231.42 | 0.00 | ORB-long ORB[1222.00,1233.00] vol=1.5x ATR=2.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-14 10:55:00 | 1236.27 | 1231.82 | 0.00 | T1 1.5R @ 1236.27 |
| Target hit | 2025-11-14 14:35:00 | 1236.00 | 1236.21 | 0.00 | Trail-exit close<VWAP |

### Cycle 39 — BUY (started 2025-11-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 10:25:00 | 1257.00 | 1252.97 | 0.00 | ORB-long ORB[1240.40,1251.10] vol=1.5x ATR=3.05 |
| Stop hit — per-position SL triggered | 2025-11-17 10:50:00 | 1253.95 | 1253.23 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-11-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-18 10:00:00 | 1258.90 | 1256.19 | 0.00 | ORB-long ORB[1247.60,1258.40] vol=5.1x ATR=2.63 |
| Stop hit — per-position SL triggered | 2025-11-18 10:05:00 | 1256.27 | 1256.23 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-11-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 09:50:00 | 1266.70 | 1277.50 | 0.00 | ORB-short ORB[1271.20,1284.60] vol=1.9x ATR=3.64 |
| Stop hit — per-position SL triggered | 2025-11-20 10:15:00 | 1270.34 | 1275.91 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-11-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-24 10:30:00 | 1288.90 | 1281.19 | 0.00 | ORB-long ORB[1273.50,1281.30] vol=1.9x ATR=3.00 |
| Stop hit — per-position SL triggered | 2025-11-24 10:50:00 | 1285.90 | 1282.62 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-12-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 09:30:00 | 1267.00 | 1271.51 | 0.00 | ORB-short ORB[1268.90,1281.00] vol=1.6x ATR=2.73 |
| Stop hit — per-position SL triggered | 2025-12-02 10:20:00 | 1269.73 | 1269.02 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-12-05 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 10:30:00 | 1271.40 | 1276.32 | 0.00 | ORB-short ORB[1274.40,1280.30] vol=2.5x ATR=3.17 |
| Stop hit — per-position SL triggered | 2025-12-05 11:45:00 | 1274.57 | 1273.66 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-12-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 11:10:00 | 1266.80 | 1274.15 | 0.00 | ORB-short ORB[1277.70,1288.80] vol=2.4x ATR=2.04 |
| Stop hit — per-position SL triggered | 2025-12-08 11:25:00 | 1268.84 | 1273.66 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-12-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 09:50:00 | 1229.90 | 1226.33 | 0.00 | ORB-long ORB[1220.20,1228.80] vol=1.6x ATR=2.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 10:15:00 | 1233.60 | 1228.21 | 0.00 | T1 1.5R @ 1233.60 |
| Target hit | 2025-12-18 13:55:00 | 1231.20 | 1232.32 | 0.00 | Trail-exit close<VWAP |

### Cycle 47 — SELL (started 2025-12-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-22 11:00:00 | 1228.10 | 1230.72 | 0.00 | ORB-short ORB[1231.30,1236.90] vol=1.7x ATR=1.93 |
| Stop hit — per-position SL triggered | 2025-12-22 11:20:00 | 1230.03 | 1230.37 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-12-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-23 09:55:00 | 1230.80 | 1232.60 | 0.00 | ORB-short ORB[1233.00,1239.00] vol=2.6x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 10:20:00 | 1228.53 | 1231.87 | 0.00 | T1 1.5R @ 1228.53 |
| Stop hit — per-position SL triggered | 2025-12-23 10:45:00 | 1230.80 | 1231.24 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-12-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 09:50:00 | 1220.10 | 1222.35 | 0.00 | ORB-short ORB[1221.00,1231.60] vol=2.1x ATR=2.08 |
| Stop hit — per-position SL triggered | 2025-12-29 10:15:00 | 1222.18 | 1221.70 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-12-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 10:10:00 | 1240.40 | 1235.63 | 0.00 | ORB-long ORB[1232.00,1237.00] vol=1.6x ATR=2.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 11:40:00 | 1243.42 | 1238.01 | 0.00 | T1 1.5R @ 1243.42 |
| Target hit | 2025-12-30 15:20:00 | 1246.40 | 1244.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — BUY (started 2026-01-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 10:50:00 | 1274.20 | 1269.51 | 0.00 | ORB-long ORB[1263.80,1271.10] vol=3.3x ATR=2.59 |
| Stop hit — per-position SL triggered | 2026-01-01 11:45:00 | 1271.61 | 1270.49 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2026-01-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:45:00 | 1279.20 | 1276.98 | 0.00 | ORB-long ORB[1271.00,1278.30] vol=2.3x ATR=2.10 |
| Stop hit — per-position SL triggered | 2026-01-02 10:55:00 | 1277.10 | 1277.17 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2026-01-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 10:40:00 | 1268.50 | 1273.68 | 0.00 | ORB-short ORB[1271.10,1284.00] vol=1.5x ATR=2.44 |
| Stop hit — per-position SL triggered | 2026-01-13 11:15:00 | 1270.94 | 1273.01 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2026-01-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 10:50:00 | 1293.20 | 1276.24 | 0.00 | ORB-long ORB[1253.50,1265.80] vol=2.9x ATR=4.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 11:05:00 | 1299.51 | 1278.42 | 0.00 | T1 1.5R @ 1299.51 |
| Stop hit — per-position SL triggered | 2026-01-14 13:35:00 | 1293.20 | 1287.22 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2026-01-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 10:55:00 | 1298.40 | 1304.06 | 0.00 | ORB-short ORB[1302.90,1310.90] vol=2.7x ATR=2.76 |
| Stop hit — per-position SL triggered | 2026-01-20 11:10:00 | 1301.16 | 1303.37 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2026-01-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 10:55:00 | 1282.70 | 1290.88 | 0.00 | ORB-short ORB[1286.00,1297.80] vol=1.6x ATR=3.30 |
| Stop hit — per-position SL triggered | 2026-01-22 11:25:00 | 1286.00 | 1289.74 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2026-01-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-23 11:05:00 | 1282.70 | 1287.87 | 0.00 | ORB-short ORB[1285.60,1299.00] vol=1.6x ATR=3.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 11:35:00 | 1278.01 | 1286.91 | 0.00 | T1 1.5R @ 1278.01 |
| Target hit | 2026-01-23 15:20:00 | 1256.10 | 1271.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 58 — BUY (started 2026-01-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 10:45:00 | 1374.80 | 1367.21 | 0.00 | ORB-long ORB[1352.00,1367.70] vol=1.7x ATR=3.84 |
| Stop hit — per-position SL triggered | 2026-01-30 11:15:00 | 1370.96 | 1368.39 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2026-02-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-01 11:05:00 | 1365.00 | 1370.13 | 0.00 | ORB-short ORB[1372.10,1384.90] vol=5.3x ATR=3.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 11:45:00 | 1359.80 | 1367.74 | 0.00 | T1 1.5R @ 1359.80 |
| Stop hit — per-position SL triggered | 2026-02-01 11:55:00 | 1365.00 | 1367.54 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2026-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-04 11:15:00 | 1347.70 | 1356.84 | 0.00 | ORB-short ORB[1353.00,1369.90] vol=1.7x ATR=2.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-04 11:25:00 | 1343.32 | 1355.41 | 0.00 | T1 1.5R @ 1343.32 |
| Stop hit — per-position SL triggered | 2026-02-04 11:30:00 | 1347.70 | 1355.14 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2026-02-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:55:00 | 1324.70 | 1337.66 | 0.00 | ORB-short ORB[1339.30,1346.50] vol=1.6x ATR=3.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 10:10:00 | 1319.63 | 1334.33 | 0.00 | T1 1.5R @ 1319.63 |
| Stop hit — per-position SL triggered | 2026-02-13 10:25:00 | 1324.70 | 1332.49 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2026-02-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 11:05:00 | 1359.80 | 1355.15 | 0.00 | ORB-long ORB[1351.40,1358.30] vol=3.2x ATR=2.43 |
| Stop hit — per-position SL triggered | 2026-02-17 11:30:00 | 1357.37 | 1355.40 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2026-02-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 11:00:00 | 1366.50 | 1360.04 | 0.00 | ORB-long ORB[1355.50,1364.00] vol=2.2x ATR=2.43 |
| Stop hit — per-position SL triggered | 2026-02-18 11:10:00 | 1364.07 | 1360.44 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-02-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:55:00 | 1395.70 | 1393.60 | 0.00 | ORB-long ORB[1387.60,1395.00] vol=7.2x ATR=2.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:25:00 | 1399.36 | 1394.18 | 0.00 | T1 1.5R @ 1399.36 |
| Stop hit — per-position SL triggered | 2026-02-25 12:35:00 | 1395.70 | 1395.05 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2026-03-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 09:45:00 | 1332.90 | 1337.78 | 0.00 | ORB-short ORB[1335.80,1347.90] vol=1.5x ATR=4.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 09:55:00 | 1325.75 | 1334.23 | 0.00 | T1 1.5R @ 1325.75 |
| Target hit | 2026-03-06 12:05:00 | 1326.10 | 1325.79 | 0.00 | Trail-exit close>VWAP |

### Cycle 66 — BUY (started 2026-03-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 11:10:00 | 1309.80 | 1298.61 | 0.00 | ORB-long ORB[1284.60,1298.50] vol=2.7x ATR=3.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 11:35:00 | 1315.16 | 1300.89 | 0.00 | T1 1.5R @ 1315.16 |
| Stop hit — per-position SL triggered | 2026-03-10 12:20:00 | 1309.80 | 1302.84 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2026-03-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 11:00:00 | 1289.90 | 1295.46 | 0.00 | ORB-short ORB[1301.40,1317.50] vol=1.7x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:10:00 | 1285.46 | 1293.97 | 0.00 | T1 1.5R @ 1285.46 |
| Target hit | 2026-03-11 15:20:00 | 1254.50 | 1274.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 68 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 11:15:00 | 1208.20 | 1218.62 | 0.00 | ORB-short ORB[1222.90,1234.50] vol=1.6x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 11:55:00 | 1203.77 | 1215.93 | 0.00 | T1 1.5R @ 1203.77 |
| Target hit | 2026-03-13 15:20:00 | 1196.60 | 1206.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 69 — SELL (started 2026-03-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 09:40:00 | 1174.40 | 1179.98 | 0.00 | ORB-short ORB[1180.00,1189.80] vol=2.3x ATR=4.71 |
| Stop hit — per-position SL triggered | 2026-03-24 10:15:00 | 1179.11 | 1178.72 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-06 11:15:00 | 1214.90 | 1206.32 | 0.00 | ORB-long ORB[1196.40,1211.30] vol=2.5x ATR=4.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-06 11:45:00 | 1221.26 | 1207.83 | 0.00 | T1 1.5R @ 1221.26 |
| Target hit | 2026-04-06 15:20:00 | 1242.70 | 1223.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 71 — BUY (started 2026-04-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 10:55:00 | 1324.20 | 1314.24 | 0.00 | ORB-long ORB[1300.10,1318.10] vol=1.6x ATR=5.21 |
| Stop hit — per-position SL triggered | 2026-04-08 13:05:00 | 1318.99 | 1317.40 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2026-04-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:55:00 | 1339.10 | 1329.70 | 0.00 | ORB-long ORB[1315.30,1330.50] vol=1.9x ATR=3.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 11:45:00 | 1344.49 | 1331.61 | 0.00 | T1 1.5R @ 1344.49 |
| Target hit | 2026-04-13 15:20:00 | 1353.30 | 1346.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — SELL (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 11:15:00 | 1358.90 | 1367.64 | 0.00 | ORB-short ORB[1364.10,1384.50] vol=1.7x ATR=3.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 12:25:00 | 1353.94 | 1365.61 | 0.00 | T1 1.5R @ 1353.94 |
| Stop hit — per-position SL triggered | 2026-04-15 14:55:00 | 1358.90 | 1360.52 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-04-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 11:00:00 | 1351.90 | 1353.51 | 0.00 | ORB-short ORB[1353.30,1364.50] vol=1.5x ATR=3.61 |
| Stop hit — per-position SL triggered | 2026-04-16 11:15:00 | 1355.51 | 1353.59 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-04-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:50:00 | 1355.50 | 1362.68 | 0.00 | ORB-short ORB[1366.00,1375.00] vol=2.3x ATR=3.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:00:00 | 1350.71 | 1362.08 | 0.00 | T1 1.5R @ 1350.71 |
| Stop hit — per-position SL triggered | 2026-04-24 11:20:00 | 1355.50 | 1360.81 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2026-04-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:05:00 | 1301.80 | 1310.03 | 0.00 | ORB-short ORB[1310.20,1319.70] vol=3.6x ATR=2.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:20:00 | 1297.97 | 1308.97 | 0.00 | T1 1.5R @ 1297.97 |
| Target hit | 2026-04-28 15:20:00 | 1289.40 | 1297.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 77 — SELL (started 2026-05-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:05:00 | 1261.70 | 1266.39 | 0.00 | ORB-short ORB[1263.10,1271.50] vol=1.9x ATR=2.93 |
| Stop hit — per-position SL triggered | 2026-05-05 12:50:00 | 1264.63 | 1264.77 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-05-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:55:00 | 1265.90 | 1271.97 | 0.00 | ORB-short ORB[1269.40,1286.30] vol=1.5x ATR=3.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 11:15:00 | 1260.95 | 1271.13 | 0.00 | T1 1.5R @ 1260.95 |
| Stop hit — per-position SL triggered | 2026-05-06 13:30:00 | 1265.90 | 1266.48 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-16 09:30:00 | 1203.70 | 2025-05-16 09:40:00 | 1207.27 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-05-21 11:05:00 | 1201.70 | 2025-05-21 11:10:00 | 1199.28 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-05-23 10:45:00 | 1205.90 | 2025-05-23 10:50:00 | 1209.91 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-05-23 10:45:00 | 1205.90 | 2025-05-23 11:45:00 | 1205.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-03 09:30:00 | 1185.00 | 2025-06-03 09:35:00 | 1187.73 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-06-04 10:50:00 | 1175.70 | 2025-06-04 11:15:00 | 1172.58 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-06-04 10:50:00 | 1175.70 | 2025-06-04 11:40:00 | 1175.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-06 10:10:00 | 1153.50 | 2025-06-06 10:20:00 | 1156.17 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-07-11 09:40:00 | 1175.60 | 2025-07-11 09:45:00 | 1173.02 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-29 10:15:00 | 1062.00 | 2025-07-29 10:20:00 | 1064.99 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-07-30 09:40:00 | 1073.50 | 2025-07-30 09:45:00 | 1070.17 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-08-04 09:45:00 | 1065.40 | 2025-08-04 10:00:00 | 1063.15 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-08-08 10:15:00 | 1063.80 | 2025-08-08 11:00:00 | 1060.47 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-08-08 10:15:00 | 1063.80 | 2025-08-08 12:55:00 | 1063.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-11 10:00:00 | 1061.00 | 2025-08-11 10:25:00 | 1065.65 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-08-11 10:00:00 | 1061.00 | 2025-08-11 15:20:00 | 1071.70 | TARGET_HIT | 0.50 | 1.01% |
| BUY | retest1 | 2025-08-12 10:15:00 | 1075.10 | 2025-08-12 10:25:00 | 1073.41 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-08-14 11:15:00 | 1068.30 | 2025-08-14 11:20:00 | 1070.07 | PARTIAL | 0.50 | 0.17% |
| BUY | retest1 | 2025-08-14 11:15:00 | 1068.30 | 2025-08-14 11:25:00 | 1068.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-19 11:15:00 | 1089.80 | 2025-08-19 11:30:00 | 1092.33 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2025-08-19 11:15:00 | 1089.80 | 2025-08-19 13:00:00 | 1089.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-22 10:35:00 | 1071.50 | 2025-08-22 10:45:00 | 1073.06 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-08-26 10:20:00 | 1061.40 | 2025-08-26 10:45:00 | 1058.87 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-08-26 10:20:00 | 1061.40 | 2025-08-26 15:20:00 | 1050.30 | TARGET_HIT | 0.50 | 1.05% |
| BUY | retest1 | 2025-08-29 11:00:00 | 1058.20 | 2025-08-29 12:35:00 | 1056.50 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-09-08 10:25:00 | 1062.20 | 2025-09-08 10:35:00 | 1060.17 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-09-09 10:40:00 | 1050.40 | 2025-09-09 11:00:00 | 1051.96 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-09-10 11:10:00 | 1069.90 | 2025-09-10 11:25:00 | 1067.90 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-09-11 11:10:00 | 1074.20 | 2025-09-11 11:25:00 | 1076.84 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2025-09-11 11:10:00 | 1074.20 | 2025-09-11 11:35:00 | 1074.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-12 09:45:00 | 1105.00 | 2025-09-12 10:15:00 | 1101.70 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-09-16 09:30:00 | 1116.20 | 2025-09-16 09:35:00 | 1119.07 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-09-16 09:30:00 | 1116.20 | 2025-09-16 09:45:00 | 1116.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-24 11:00:00 | 1158.60 | 2025-09-24 11:20:00 | 1160.87 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-09-30 10:50:00 | 1129.90 | 2025-09-30 11:40:00 | 1126.74 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-09-30 10:50:00 | 1129.90 | 2025-09-30 12:00:00 | 1129.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-01 11:15:00 | 1151.00 | 2025-10-01 11:25:00 | 1156.29 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-10-01 11:15:00 | 1151.00 | 2025-10-01 15:20:00 | 1159.90 | TARGET_HIT | 0.50 | 0.77% |
| SELL | retest1 | 2025-10-07 11:10:00 | 1194.80 | 2025-10-07 11:20:00 | 1190.08 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-10-07 11:10:00 | 1194.80 | 2025-10-07 13:00:00 | 1194.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-09 10:00:00 | 1175.70 | 2025-10-09 10:15:00 | 1171.40 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-10-09 10:00:00 | 1175.70 | 2025-10-09 11:35:00 | 1173.90 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2025-10-10 09:35:00 | 1178.50 | 2025-10-10 09:45:00 | 1182.76 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-10-10 09:35:00 | 1178.50 | 2025-10-10 14:15:00 | 1183.60 | TARGET_HIT | 0.50 | 0.43% |
| BUY | retest1 | 2025-10-23 09:40:00 | 1262.60 | 2025-10-23 10:05:00 | 1268.68 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-10-23 09:40:00 | 1262.60 | 2025-10-23 12:45:00 | 1265.60 | TARGET_HIT | 0.50 | 0.24% |
| SELL | retest1 | 2025-10-24 11:05:00 | 1245.70 | 2025-10-24 11:10:00 | 1248.24 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-10-28 10:10:00 | 1245.40 | 2025-10-28 10:45:00 | 1248.47 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-10-29 10:15:00 | 1240.30 | 2025-10-29 10:25:00 | 1242.77 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-11-03 09:35:00 | 1231.00 | 2025-11-03 09:40:00 | 1228.21 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-11-10 09:35:00 | 1231.20 | 2025-11-10 09:40:00 | 1228.57 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-11-13 10:50:00 | 1228.50 | 2025-11-13 11:15:00 | 1226.32 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-11-14 10:50:00 | 1233.10 | 2025-11-14 10:55:00 | 1236.27 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-11-14 10:50:00 | 1233.10 | 2025-11-14 14:35:00 | 1236.00 | TARGET_HIT | 0.50 | 0.24% |
| BUY | retest1 | 2025-11-17 10:25:00 | 1257.00 | 2025-11-17 10:50:00 | 1253.95 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-11-18 10:00:00 | 1258.90 | 2025-11-18 10:05:00 | 1256.27 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-11-20 09:50:00 | 1266.70 | 2025-11-20 10:15:00 | 1270.34 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-11-24 10:30:00 | 1288.90 | 2025-11-24 10:50:00 | 1285.90 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-12-02 09:30:00 | 1267.00 | 2025-12-02 10:20:00 | 1269.73 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-12-05 10:30:00 | 1271.40 | 2025-12-05 11:45:00 | 1274.57 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-12-08 11:10:00 | 1266.80 | 2025-12-08 11:25:00 | 1268.84 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-12-18 09:50:00 | 1229.90 | 2025-12-18 10:15:00 | 1233.60 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-12-18 09:50:00 | 1229.90 | 2025-12-18 13:55:00 | 1231.20 | TARGET_HIT | 0.50 | 0.11% |
| SELL | retest1 | 2025-12-22 11:00:00 | 1228.10 | 2025-12-22 11:20:00 | 1230.03 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-12-23 09:55:00 | 1230.80 | 2025-12-23 10:20:00 | 1228.53 | PARTIAL | 0.50 | 0.18% |
| SELL | retest1 | 2025-12-23 09:55:00 | 1230.80 | 2025-12-23 10:45:00 | 1230.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-29 09:50:00 | 1220.10 | 2025-12-29 10:15:00 | 1222.18 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-12-30 10:10:00 | 1240.40 | 2025-12-30 11:40:00 | 1243.42 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2025-12-30 10:10:00 | 1240.40 | 2025-12-30 15:20:00 | 1246.40 | TARGET_HIT | 0.50 | 0.48% |
| BUY | retest1 | 2026-01-01 10:50:00 | 1274.20 | 2026-01-01 11:45:00 | 1271.61 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-01-02 10:45:00 | 1279.20 | 2026-01-02 10:55:00 | 1277.10 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2026-01-13 10:40:00 | 1268.50 | 2026-01-13 11:15:00 | 1270.94 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-01-14 10:50:00 | 1293.20 | 2026-01-14 11:05:00 | 1299.51 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-01-14 10:50:00 | 1293.20 | 2026-01-14 13:35:00 | 1293.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-20 10:55:00 | 1298.40 | 2026-01-20 11:10:00 | 1301.16 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-01-22 10:55:00 | 1282.70 | 2026-01-22 11:25:00 | 1286.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-01-23 11:05:00 | 1282.70 | 2026-01-23 11:35:00 | 1278.01 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-01-23 11:05:00 | 1282.70 | 2026-01-23 15:20:00 | 1256.10 | TARGET_HIT | 0.50 | 2.07% |
| BUY | retest1 | 2026-01-30 10:45:00 | 1374.80 | 2026-01-30 11:15:00 | 1370.96 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-01 11:05:00 | 1365.00 | 2026-02-01 11:45:00 | 1359.80 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-02-01 11:05:00 | 1365.00 | 2026-02-01 11:55:00 | 1365.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-04 11:15:00 | 1347.70 | 2026-02-04 11:25:00 | 1343.32 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-02-04 11:15:00 | 1347.70 | 2026-02-04 11:30:00 | 1347.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-13 09:55:00 | 1324.70 | 2026-02-13 10:10:00 | 1319.63 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-02-13 09:55:00 | 1324.70 | 2026-02-13 10:25:00 | 1324.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 11:05:00 | 1359.80 | 2026-02-17 11:30:00 | 1357.37 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2026-02-18 11:00:00 | 1366.50 | 2026-02-18 11:10:00 | 1364.07 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2026-02-25 10:55:00 | 1395.70 | 2026-02-25 11:25:00 | 1399.36 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2026-02-25 10:55:00 | 1395.70 | 2026-02-25 12:35:00 | 1395.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-06 09:45:00 | 1332.90 | 2026-03-06 09:55:00 | 1325.75 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-03-06 09:45:00 | 1332.90 | 2026-03-06 12:05:00 | 1326.10 | TARGET_HIT | 0.50 | 0.51% |
| BUY | retest1 | 2026-03-10 11:10:00 | 1309.80 | 2026-03-10 11:35:00 | 1315.16 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-03-10 11:10:00 | 1309.80 | 2026-03-10 12:20:00 | 1309.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-11 11:00:00 | 1289.90 | 2026-03-11 11:10:00 | 1285.46 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-03-11 11:00:00 | 1289.90 | 2026-03-11 15:20:00 | 1254.50 | TARGET_HIT | 0.50 | 2.74% |
| SELL | retest1 | 2026-03-13 11:15:00 | 1208.20 | 2026-03-13 11:55:00 | 1203.77 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-03-13 11:15:00 | 1208.20 | 2026-03-13 15:20:00 | 1196.60 | TARGET_HIT | 0.50 | 0.96% |
| SELL | retest1 | 2026-03-24 09:40:00 | 1174.40 | 2026-03-24 10:15:00 | 1179.11 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-06 11:15:00 | 1214.90 | 2026-04-06 11:45:00 | 1221.26 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-04-06 11:15:00 | 1214.90 | 2026-04-06 15:20:00 | 1242.70 | TARGET_HIT | 0.50 | 2.29% |
| BUY | retest1 | 2026-04-08 10:55:00 | 1324.20 | 2026-04-08 13:05:00 | 1318.99 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-04-13 10:55:00 | 1339.10 | 2026-04-13 11:45:00 | 1344.49 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-04-13 10:55:00 | 1339.10 | 2026-04-13 15:20:00 | 1353.30 | TARGET_HIT | 0.50 | 1.06% |
| SELL | retest1 | 2026-04-15 11:15:00 | 1358.90 | 2026-04-15 12:25:00 | 1353.94 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-04-15 11:15:00 | 1358.90 | 2026-04-15 14:55:00 | 1358.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-16 11:00:00 | 1351.90 | 2026-04-16 11:15:00 | 1355.51 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-04-24 10:50:00 | 1355.50 | 2026-04-24 11:00:00 | 1350.71 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-04-24 10:50:00 | 1355.50 | 2026-04-24 11:20:00 | 1355.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-28 11:05:00 | 1301.80 | 2026-04-28 11:20:00 | 1297.97 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2026-04-28 11:05:00 | 1301.80 | 2026-04-28 15:20:00 | 1289.40 | TARGET_HIT | 0.50 | 0.95% |
| SELL | retest1 | 2026-05-05 11:05:00 | 1261.70 | 2026-05-05 12:50:00 | 1264.63 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-05-06 10:55:00 | 1265.90 | 2026-05-06 11:15:00 | 1260.95 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-05-06 10:55:00 | 1265.90 | 2026-05-06 13:30:00 | 1265.90 | STOP_HIT | 0.50 | 0.00% |
