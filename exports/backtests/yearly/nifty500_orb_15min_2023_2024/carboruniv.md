# Carborundum Universal Ltd. (CARBORUNIV)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (53705 bars)
- **Last close:** 1020.20
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
| ENTRY1 | 95 |
| ENTRY2 | 0 |
| PARTIAL | 38 |
| TARGET_HIT | 24 |
| STOP_HIT | 71 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 133 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 62 / 71
- **Target hits / Stop hits / Partials:** 24 / 71 / 38
- **Avg / median % per leg:** 0.26% / 0.00%
- **Sum % (uncompounded):** 34.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 59 | 24 | 40.7% | 10 | 35 | 14 | 0.18% | 10.3% |
| BUY @ 2nd Alert (retest1) | 59 | 24 | 40.7% | 10 | 35 | 14 | 0.18% | 10.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 74 | 38 | 51.4% | 14 | 36 | 24 | 0.32% | 23.9% |
| SELL @ 2nd Alert (retest1) | 74 | 38 | 51.4% | 14 | 36 | 24 | 0.32% | 23.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 133 | 62 | 46.6% | 24 | 71 | 38 | 0.26% | 34.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-16 10:45:00 | 1149.60 | 1141.24 | 0.00 | ORB-long ORB[1135.15,1146.90] vol=2.0x ATR=3.27 |
| Stop hit — per-position SL triggered | 2023-05-16 10:50:00 | 1146.33 | 1141.43 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-18 10:45:00 | 1167.20 | 1169.59 | 0.00 | ORB-short ORB[1170.00,1178.15] vol=1.8x ATR=2.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-18 11:15:00 | 1162.80 | 1169.03 | 0.00 | T1 1.5R @ 1162.80 |
| Stop hit — per-position SL triggered | 2023-05-18 12:25:00 | 1167.20 | 1166.96 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-05-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-19 10:25:00 | 1183.05 | 1173.72 | 0.00 | ORB-long ORB[1163.15,1175.00] vol=6.3x ATR=4.81 |
| Stop hit — per-position SL triggered | 2023-05-19 10:35:00 | 1178.24 | 1174.80 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-05-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-23 11:05:00 | 1222.45 | 1228.75 | 0.00 | ORB-short ORB[1223.50,1235.00] vol=2.5x ATR=3.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-23 11:45:00 | 1217.66 | 1227.07 | 0.00 | T1 1.5R @ 1217.66 |
| Target hit | 2023-05-23 15:20:00 | 1196.00 | 1216.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2023-05-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-24 10:00:00 | 1179.00 | 1184.17 | 0.00 | ORB-short ORB[1180.35,1197.90] vol=2.4x ATR=4.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-24 10:10:00 | 1172.46 | 1182.00 | 0.00 | T1 1.5R @ 1172.46 |
| Stop hit — per-position SL triggered | 2023-05-24 10:30:00 | 1179.00 | 1179.99 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-05-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-25 10:50:00 | 1180.60 | 1172.22 | 0.00 | ORB-long ORB[1166.00,1179.95] vol=1.9x ATR=4.10 |
| Stop hit — per-position SL triggered | 2023-05-25 11:00:00 | 1176.50 | 1173.52 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-05-31 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-31 10:45:00 | 1161.60 | 1148.21 | 0.00 | ORB-long ORB[1140.00,1154.05] vol=1.6x ATR=3.46 |
| Stop hit — per-position SL triggered | 2023-05-31 11:00:00 | 1158.14 | 1150.91 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-06-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-09 09:45:00 | 1192.05 | 1196.81 | 0.00 | ORB-short ORB[1195.60,1200.00] vol=2.0x ATR=3.44 |
| Stop hit — per-position SL triggered | 2023-06-09 09:55:00 | 1195.49 | 1196.21 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-06-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-12 09:35:00 | 1199.70 | 1190.92 | 0.00 | ORB-long ORB[1176.05,1193.70] vol=1.7x ATR=5.31 |
| Stop hit — per-position SL triggered | 2023-06-12 15:10:00 | 1194.39 | 1197.56 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-06-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-13 09:55:00 | 1199.85 | 1194.87 | 0.00 | ORB-long ORB[1187.65,1196.70] vol=2.1x ATR=3.61 |
| Stop hit — per-position SL triggered | 2023-06-13 10:50:00 | 1196.24 | 1197.92 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-06-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-14 10:50:00 | 1201.30 | 1200.17 | 0.00 | ORB-long ORB[1192.00,1200.00] vol=12.8x ATR=3.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-14 11:00:00 | 1205.89 | 1201.55 | 0.00 | T1 1.5R @ 1205.89 |
| Target hit | 2023-06-14 14:25:00 | 1202.65 | 1205.47 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — SELL (started 2023-06-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-21 11:10:00 | 1204.15 | 1214.85 | 0.00 | ORB-short ORB[1210.00,1220.00] vol=1.7x ATR=3.17 |
| Stop hit — per-position SL triggered | 2023-06-21 11:35:00 | 1207.32 | 1213.49 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-06-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-22 10:00:00 | 1233.25 | 1222.16 | 0.00 | ORB-long ORB[1216.20,1225.00] vol=5.0x ATR=3.43 |
| Stop hit — per-position SL triggered | 2023-06-22 10:05:00 | 1229.82 | 1223.46 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-06-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-26 09:40:00 | 1205.90 | 1202.67 | 0.00 | ORB-long ORB[1191.15,1200.70] vol=2.0x ATR=4.32 |
| Stop hit — per-position SL triggered | 2023-06-26 09:50:00 | 1201.58 | 1202.67 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-06-28 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-28 10:25:00 | 1162.25 | 1171.14 | 0.00 | ORB-short ORB[1168.00,1182.85] vol=1.6x ATR=4.85 |
| Stop hit — per-position SL triggered | 2023-06-28 11:10:00 | 1167.10 | 1169.24 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2023-06-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-30 09:45:00 | 1191.15 | 1197.73 | 0.00 | ORB-short ORB[1194.45,1204.20] vol=1.5x ATR=4.83 |
| Stop hit — per-position SL triggered | 2023-06-30 09:50:00 | 1195.98 | 1197.19 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-07-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-04 09:40:00 | 1224.10 | 1221.20 | 0.00 | ORB-long ORB[1203.45,1220.00] vol=9.7x ATR=4.62 |
| Stop hit — per-position SL triggered | 2023-07-04 09:45:00 | 1219.48 | 1220.55 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-07-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-07 10:40:00 | 1210.45 | 1201.70 | 0.00 | ORB-long ORB[1195.05,1208.80] vol=2.8x ATR=4.07 |
| Stop hit — per-position SL triggered | 2023-07-07 11:15:00 | 1206.38 | 1206.94 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-10 10:15:00 | 1184.50 | 1194.56 | 0.00 | ORB-short ORB[1193.20,1209.15] vol=1.6x ATR=5.15 |
| Stop hit — per-position SL triggered | 2023-07-10 10:30:00 | 1189.65 | 1193.07 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2023-07-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-13 10:30:00 | 1193.30 | 1195.51 | 0.00 | ORB-short ORB[1193.55,1204.00] vol=3.4x ATR=4.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-13 10:50:00 | 1187.02 | 1193.64 | 0.00 | T1 1.5R @ 1187.02 |
| Stop hit — per-position SL triggered | 2023-07-13 12:20:00 | 1193.30 | 1190.30 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-14 10:15:00 | 1176.55 | 1184.89 | 0.00 | ORB-short ORB[1184.85,1192.85] vol=1.8x ATR=4.06 |
| Stop hit — per-position SL triggered | 2023-07-14 11:05:00 | 1180.61 | 1179.29 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-07-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-17 10:50:00 | 1193.15 | 1196.00 | 0.00 | ORB-short ORB[1194.15,1202.40] vol=5.5x ATR=5.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-17 13:05:00 | 1185.49 | 1193.85 | 0.00 | T1 1.5R @ 1185.49 |
| Target hit | 2023-07-17 15:20:00 | 1177.05 | 1184.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — BUY (started 2023-07-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-18 09:40:00 | 1192.05 | 1188.38 | 0.00 | ORB-long ORB[1176.30,1190.55] vol=1.9x ATR=5.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-18 09:50:00 | 1200.21 | 1192.39 | 0.00 | T1 1.5R @ 1200.21 |
| Target hit | 2023-07-18 10:30:00 | 1194.40 | 1195.43 | 0.00 | Trail-exit close<VWAP |

### Cycle 24 — BUY (started 2023-07-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-19 11:10:00 | 1218.05 | 1212.23 | 0.00 | ORB-long ORB[1183.90,1202.20] vol=3.1x ATR=4.84 |
| Stop hit — per-position SL triggered | 2023-07-19 12:20:00 | 1213.21 | 1214.07 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2023-07-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-21 10:40:00 | 1199.90 | 1205.78 | 0.00 | ORB-short ORB[1210.50,1219.50] vol=4.3x ATR=3.72 |
| Stop hit — per-position SL triggered | 2023-07-21 11:45:00 | 1203.62 | 1204.80 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-07-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-25 10:35:00 | 1190.20 | 1197.06 | 0.00 | ORB-short ORB[1191.05,1202.15] vol=1.8x ATR=4.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-25 11:05:00 | 1183.60 | 1193.78 | 0.00 | T1 1.5R @ 1183.60 |
| Target hit | 2023-07-25 14:50:00 | 1185.00 | 1184.77 | 0.00 | Trail-exit close>VWAP |

### Cycle 27 — BUY (started 2023-07-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-27 10:20:00 | 1212.80 | 1205.99 | 0.00 | ORB-long ORB[1195.05,1209.50] vol=1.7x ATR=5.07 |
| Stop hit — per-position SL triggered | 2023-07-27 11:00:00 | 1207.73 | 1209.29 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2023-08-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-16 09:40:00 | 1049.80 | 1054.44 | 0.00 | ORB-short ORB[1052.00,1063.15] vol=1.9x ATR=3.92 |
| Stop hit — per-position SL triggered | 2023-08-16 09:45:00 | 1053.72 | 1054.37 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-08-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-17 09:35:00 | 1085.80 | 1087.85 | 0.00 | ORB-short ORB[1086.05,1101.85] vol=5.8x ATR=5.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-17 09:40:00 | 1077.91 | 1086.80 | 0.00 | T1 1.5R @ 1077.91 |
| Target hit | 2023-08-17 11:30:00 | 1072.30 | 1072.09 | 0.00 | Trail-exit close>VWAP |

### Cycle 30 — BUY (started 2023-08-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-18 10:10:00 | 1091.00 | 1084.99 | 0.00 | ORB-long ORB[1067.90,1079.50] vol=2.6x ATR=5.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-18 11:40:00 | 1099.75 | 1088.27 | 0.00 | T1 1.5R @ 1099.75 |
| Stop hit — per-position SL triggered | 2023-08-18 13:25:00 | 1091.00 | 1089.74 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2023-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-23 10:15:00 | 1129.70 | 1122.64 | 0.00 | ORB-long ORB[1113.05,1125.30] vol=2.0x ATR=3.94 |
| Stop hit — per-position SL triggered | 2023-08-23 10:20:00 | 1125.76 | 1123.08 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-08-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-24 09:30:00 | 1132.05 | 1140.37 | 0.00 | ORB-short ORB[1140.00,1149.20] vol=1.8x ATR=4.99 |
| Stop hit — per-position SL triggered | 2023-08-24 09:45:00 | 1137.04 | 1139.14 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2023-08-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-25 09:40:00 | 1126.00 | 1131.89 | 0.00 | ORB-short ORB[1130.00,1143.90] vol=3.3x ATR=3.51 |
| Stop hit — per-position SL triggered | 2023-08-25 09:50:00 | 1129.51 | 1130.52 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-08-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 10:20:00 | 1142.85 | 1137.80 | 0.00 | ORB-long ORB[1135.05,1141.85] vol=1.5x ATR=2.97 |
| Stop hit — per-position SL triggered | 2023-08-30 11:00:00 | 1139.88 | 1138.67 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2023-09-01 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-01 09:55:00 | 1116.05 | 1124.12 | 0.00 | ORB-short ORB[1132.05,1141.90] vol=1.5x ATR=4.11 |
| Stop hit — per-position SL triggered | 2023-09-01 10:55:00 | 1120.16 | 1120.80 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-09-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 10:30:00 | 1147.65 | 1138.35 | 0.00 | ORB-long ORB[1131.05,1145.00] vol=2.4x ATR=4.32 |
| Stop hit — per-position SL triggered | 2023-09-04 11:40:00 | 1143.33 | 1143.86 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-09-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-08 10:40:00 | 1183.00 | 1187.40 | 0.00 | ORB-short ORB[1185.55,1199.00] vol=1.8x ATR=3.34 |
| Stop hit — per-position SL triggered | 2023-09-08 10:50:00 | 1186.34 | 1187.34 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2023-09-14 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-14 09:50:00 | 1206.25 | 1198.89 | 0.00 | ORB-long ORB[1186.10,1198.00] vol=5.0x ATR=5.21 |
| Stop hit — per-position SL triggered | 2023-09-14 10:00:00 | 1201.04 | 1199.22 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2023-09-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-15 09:45:00 | 1192.05 | 1196.34 | 0.00 | ORB-short ORB[1195.05,1205.45] vol=1.7x ATR=3.11 |
| Stop hit — per-position SL triggered | 2023-09-15 09:55:00 | 1195.16 | 1195.90 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2023-09-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-18 10:50:00 | 1180.00 | 1188.58 | 0.00 | ORB-short ORB[1184.00,1194.20] vol=4.5x ATR=3.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-18 10:55:00 | 1174.05 | 1186.39 | 0.00 | T1 1.5R @ 1174.05 |
| Stop hit — per-position SL triggered | 2023-09-18 11:05:00 | 1180.00 | 1185.80 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2023-09-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-20 10:55:00 | 1171.05 | 1176.39 | 0.00 | ORB-short ORB[1171.45,1181.35] vol=4.1x ATR=3.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-20 11:00:00 | 1165.57 | 1174.97 | 0.00 | T1 1.5R @ 1165.57 |
| Target hit | 2023-09-20 12:40:00 | 1158.55 | 1158.28 | 0.00 | Trail-exit close>VWAP |

### Cycle 42 — SELL (started 2023-09-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-21 09:45:00 | 1157.50 | 1161.98 | 0.00 | ORB-short ORB[1159.15,1169.55] vol=1.5x ATR=4.93 |
| Stop hit — per-position SL triggered | 2023-09-21 09:50:00 | 1162.43 | 1161.96 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2023-09-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-26 09:40:00 | 1184.30 | 1190.02 | 0.00 | ORB-short ORB[1191.50,1199.70] vol=2.4x ATR=4.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-26 09:55:00 | 1178.20 | 1187.27 | 0.00 | T1 1.5R @ 1178.20 |
| Target hit | 2023-09-26 14:35:00 | 1180.00 | 1176.22 | 0.00 | Trail-exit close>VWAP |

### Cycle 44 — BUY (started 2023-09-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-28 09:40:00 | 1190.40 | 1186.02 | 0.00 | ORB-long ORB[1176.55,1188.00] vol=2.2x ATR=3.81 |
| Stop hit — per-position SL triggered | 2023-09-28 12:35:00 | 1186.59 | 1188.25 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2023-10-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-04 09:40:00 | 1172.95 | 1169.26 | 0.00 | ORB-long ORB[1162.60,1169.80] vol=2.0x ATR=3.69 |
| Stop hit — per-position SL triggered | 2023-10-04 09:45:00 | 1169.26 | 1169.38 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2023-10-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-05 10:20:00 | 1180.10 | 1184.84 | 0.00 | ORB-short ORB[1183.05,1191.00] vol=2.1x ATR=4.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-05 11:40:00 | 1173.99 | 1182.02 | 0.00 | T1 1.5R @ 1173.99 |
| Stop hit — per-position SL triggered | 2023-10-05 11:45:00 | 1180.10 | 1181.91 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2023-10-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-06 10:55:00 | 1179.35 | 1184.55 | 0.00 | ORB-short ORB[1181.90,1194.00] vol=2.1x ATR=2.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-06 11:25:00 | 1174.96 | 1181.65 | 0.00 | T1 1.5R @ 1174.96 |
| Target hit | 2023-10-06 15:20:00 | 1167.85 | 1172.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — SELL (started 2023-10-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-09 10:35:00 | 1143.35 | 1153.80 | 0.00 | ORB-short ORB[1153.10,1165.00] vol=2.0x ATR=3.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-09 11:10:00 | 1137.87 | 1148.75 | 0.00 | T1 1.5R @ 1137.87 |
| Target hit | 2023-10-09 14:35:00 | 1133.60 | 1132.91 | 0.00 | Trail-exit close>VWAP |

### Cycle 49 — BUY (started 2023-10-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 09:55:00 | 1160.40 | 1156.12 | 0.00 | ORB-long ORB[1151.85,1160.00] vol=18.9x ATR=4.57 |
| Stop hit — per-position SL triggered | 2023-10-11 10:05:00 | 1155.83 | 1156.13 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2023-10-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-20 09:30:00 | 1135.10 | 1138.07 | 0.00 | ORB-short ORB[1138.05,1153.00] vol=4.2x ATR=3.85 |
| Stop hit — per-position SL triggered | 2023-10-20 09:45:00 | 1138.95 | 1137.95 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2023-11-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 11:05:00 | 1083.20 | 1080.16 | 0.00 | ORB-long ORB[1069.00,1078.50] vol=8.5x ATR=3.03 |
| Stop hit — per-position SL triggered | 2023-11-08 11:15:00 | 1080.17 | 1080.27 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2023-11-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-09 10:00:00 | 1057.05 | 1065.44 | 0.00 | ORB-short ORB[1066.00,1073.95] vol=4.3x ATR=3.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-09 10:05:00 | 1052.35 | 1057.56 | 0.00 | T1 1.5R @ 1052.35 |
| Stop hit — per-position SL triggered | 2023-11-09 10:15:00 | 1057.05 | 1056.56 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2023-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-10 10:15:00 | 1078.55 | 1075.21 | 0.00 | ORB-long ORB[1067.05,1076.75] vol=2.0x ATR=3.23 |
| Stop hit — per-position SL triggered | 2023-11-10 10:25:00 | 1075.32 | 1075.49 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-11-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-13 10:10:00 | 1073.20 | 1069.73 | 0.00 | ORB-long ORB[1066.00,1072.80] vol=5.5x ATR=2.87 |
| Stop hit — per-position SL triggered | 2023-11-13 10:15:00 | 1070.33 | 1069.77 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2023-11-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-15 10:15:00 | 1090.70 | 1087.32 | 0.00 | ORB-long ORB[1075.00,1087.10] vol=1.6x ATR=4.71 |
| Stop hit — per-position SL triggered | 2023-11-15 10:35:00 | 1085.99 | 1088.04 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2023-11-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-20 10:10:00 | 1111.45 | 1116.59 | 0.00 | ORB-short ORB[1112.00,1126.45] vol=1.6x ATR=4.20 |
| Stop hit — per-position SL triggered | 2023-11-20 10:25:00 | 1115.65 | 1116.26 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2023-11-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-22 10:45:00 | 1089.15 | 1092.23 | 0.00 | ORB-short ORB[1090.10,1100.05] vol=1.7x ATR=2.71 |
| Stop hit — per-position SL triggered | 2023-11-22 11:00:00 | 1091.86 | 1091.82 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2023-11-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-24 10:55:00 | 1137.05 | 1134.05 | 0.00 | ORB-long ORB[1128.00,1137.00] vol=8.2x ATR=2.79 |
| Stop hit — per-position SL triggered | 2023-11-24 11:30:00 | 1134.26 | 1135.14 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2023-12-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-05 11:10:00 | 1186.15 | 1195.96 | 0.00 | ORB-short ORB[1191.00,1204.95] vol=1.8x ATR=3.50 |
| Stop hit — per-position SL triggered | 2023-12-05 11:30:00 | 1189.65 | 1192.40 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2023-12-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-07 10:20:00 | 1192.55 | 1189.95 | 0.00 | ORB-long ORB[1182.55,1192.00] vol=16.0x ATR=2.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-07 10:30:00 | 1196.60 | 1190.21 | 0.00 | T1 1.5R @ 1196.60 |
| Target hit | 2023-12-07 15:20:00 | 1211.20 | 1198.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — SELL (started 2023-12-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-11 10:05:00 | 1194.05 | 1199.98 | 0.00 | ORB-short ORB[1197.10,1209.85] vol=1.7x ATR=3.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-11 10:30:00 | 1189.42 | 1196.00 | 0.00 | T1 1.5R @ 1189.42 |
| Stop hit — per-position SL triggered | 2023-12-11 10:50:00 | 1194.05 | 1194.97 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2023-12-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-12 09:50:00 | 1206.35 | 1203.09 | 0.00 | ORB-long ORB[1193.65,1201.60] vol=1.8x ATR=2.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-12 10:05:00 | 1210.28 | 1207.56 | 0.00 | T1 1.5R @ 1210.28 |
| Stop hit — per-position SL triggered | 2023-12-12 10:15:00 | 1206.35 | 1209.47 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2023-12-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-13 11:00:00 | 1207.30 | 1214.64 | 0.00 | ORB-short ORB[1216.05,1229.00] vol=1.8x ATR=3.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-13 11:30:00 | 1202.00 | 1214.07 | 0.00 | T1 1.5R @ 1202.00 |
| Target hit | 2023-12-13 15:20:00 | 1183.30 | 1190.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — BUY (started 2023-12-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 09:35:00 | 1203.50 | 1203.15 | 0.00 | ORB-long ORB[1196.75,1203.05] vol=1.6x ATR=2.85 |
| Stop hit — per-position SL triggered | 2023-12-15 09:40:00 | 1200.65 | 1203.17 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2023-12-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-19 09:40:00 | 1204.50 | 1199.65 | 0.00 | ORB-long ORB[1192.00,1198.00] vol=2.1x ATR=3.94 |
| Stop hit — per-position SL triggered | 2023-12-19 09:45:00 | 1200.56 | 1201.07 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2023-12-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-20 11:00:00 | 1163.70 | 1175.20 | 0.00 | ORB-short ORB[1172.90,1182.00] vol=7.5x ATR=4.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 11:10:00 | 1157.67 | 1170.20 | 0.00 | T1 1.5R @ 1157.67 |
| Target hit | 2023-12-20 15:20:00 | 1113.00 | 1123.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 67 — SELL (started 2023-12-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-22 10:45:00 | 1109.80 | 1119.94 | 0.00 | ORB-short ORB[1110.00,1123.20] vol=2.0x ATR=3.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-22 12:05:00 | 1104.21 | 1112.25 | 0.00 | T1 1.5R @ 1104.21 |
| Target hit | 2023-12-22 15:20:00 | 1095.00 | 1099.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 68 — SELL (started 2023-12-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-26 10:50:00 | 1088.10 | 1097.27 | 0.00 | ORB-short ORB[1100.00,1113.90] vol=4.4x ATR=3.25 |
| Stop hit — per-position SL triggered | 2023-12-26 11:20:00 | 1091.35 | 1095.75 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2023-12-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 09:30:00 | 1115.35 | 1111.67 | 0.00 | ORB-long ORB[1103.40,1113.45] vol=2.5x ATR=3.94 |
| Stop hit — per-position SL triggered | 2023-12-27 09:45:00 | 1111.41 | 1112.80 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2024-01-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-01 10:40:00 | 1113.20 | 1114.04 | 0.00 | ORB-short ORB[1115.40,1125.00] vol=3.6x ATR=3.22 |
| Stop hit — per-position SL triggered | 2024-01-01 10:50:00 | 1116.42 | 1114.08 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2024-01-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-04 10:55:00 | 1122.30 | 1126.68 | 0.00 | ORB-short ORB[1123.25,1133.00] vol=2.5x ATR=2.74 |
| Stop hit — per-position SL triggered | 2024-01-04 11:00:00 | 1125.04 | 1126.70 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2024-01-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 10:45:00 | 1125.95 | 1122.73 | 0.00 | ORB-long ORB[1117.00,1125.60] vol=2.4x ATR=2.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-05 11:00:00 | 1129.58 | 1123.62 | 0.00 | T1 1.5R @ 1129.58 |
| Target hit | 2024-01-05 14:20:00 | 1135.45 | 1135.95 | 0.00 | Trail-exit close<VWAP |

### Cycle 73 — BUY (started 2024-01-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 10:15:00 | 1156.95 | 1153.74 | 0.00 | ORB-long ORB[1147.30,1155.00] vol=1.8x ATR=3.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-11 10:25:00 | 1162.58 | 1154.71 | 0.00 | T1 1.5R @ 1162.58 |
| Stop hit — per-position SL triggered | 2024-01-11 10:30:00 | 1156.95 | 1154.88 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2024-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-12 11:15:00 | 1165.50 | 1159.39 | 0.00 | ORB-long ORB[1149.05,1165.00] vol=2.1x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-12 11:50:00 | 1169.94 | 1161.68 | 0.00 | T1 1.5R @ 1169.94 |
| Stop hit — per-position SL triggered | 2024-01-12 14:10:00 | 1165.50 | 1167.15 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2024-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-17 11:15:00 | 1141.65 | 1148.09 | 0.00 | ORB-short ORB[1144.05,1157.25] vol=2.2x ATR=3.61 |
| Stop hit — per-position SL triggered | 2024-01-17 12:45:00 | 1145.26 | 1146.86 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-01-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-18 09:35:00 | 1132.70 | 1136.13 | 0.00 | ORB-short ORB[1133.10,1144.45] vol=1.5x ATR=3.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-18 09:45:00 | 1128.07 | 1132.29 | 0.00 | T1 1.5R @ 1128.07 |
| Target hit | 2024-01-18 11:45:00 | 1129.15 | 1127.87 | 0.00 | Trail-exit close>VWAP |

### Cycle 77 — BUY (started 2024-01-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-30 11:00:00 | 1143.95 | 1143.32 | 0.00 | ORB-long ORB[1134.70,1142.90] vol=17.7x ATR=2.82 |
| Stop hit — per-position SL triggered | 2024-01-30 11:05:00 | 1141.13 | 1143.21 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2024-01-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 09:45:00 | 1131.90 | 1124.41 | 0.00 | ORB-long ORB[1112.00,1123.30] vol=2.0x ATR=4.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-31 09:55:00 | 1139.04 | 1126.08 | 0.00 | T1 1.5R @ 1139.04 |
| Target hit | 2024-01-31 13:05:00 | 1133.25 | 1134.00 | 0.00 | Trail-exit close<VWAP |

### Cycle 79 — BUY (started 2024-02-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 10:00:00 | 1137.15 | 1136.45 | 0.00 | ORB-long ORB[1129.10,1136.90] vol=17.6x ATR=3.46 |
| Stop hit — per-position SL triggered | 2024-02-02 10:10:00 | 1133.69 | 1136.44 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2024-02-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-05 10:35:00 | 1103.75 | 1112.47 | 0.00 | ORB-short ORB[1107.60,1122.80] vol=1.7x ATR=3.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-05 11:10:00 | 1097.91 | 1109.18 | 0.00 | T1 1.5R @ 1097.91 |
| Target hit | 2024-02-05 15:20:00 | 1075.00 | 1096.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 81 — BUY (started 2024-02-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-06 10:50:00 | 1109.40 | 1102.90 | 0.00 | ORB-long ORB[1087.90,1102.10] vol=2.5x ATR=3.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-06 12:10:00 | 1115.34 | 1105.92 | 0.00 | T1 1.5R @ 1115.34 |
| Target hit | 2024-02-06 15:20:00 | 1119.90 | 1114.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 82 — SELL (started 2024-02-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-09 11:05:00 | 1120.00 | 1129.73 | 0.00 | ORB-short ORB[1135.45,1150.00] vol=4.6x ATR=3.36 |
| Stop hit — per-position SL triggered | 2024-02-09 12:40:00 | 1123.36 | 1126.47 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2024-02-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-16 10:45:00 | 1120.75 | 1119.85 | 0.00 | ORB-long ORB[1110.00,1117.50] vol=1.5x ATR=4.31 |
| Stop hit — per-position SL triggered | 2024-02-16 10:55:00 | 1116.44 | 1119.77 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2024-02-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-26 09:50:00 | 1079.30 | 1085.08 | 0.00 | ORB-short ORB[1085.05,1097.60] vol=1.6x ATR=3.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-26 10:00:00 | 1074.59 | 1081.99 | 0.00 | T1 1.5R @ 1074.59 |
| Stop hit — per-position SL triggered | 2024-02-26 10:20:00 | 1079.30 | 1079.74 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2024-02-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-27 09:45:00 | 1059.15 | 1063.90 | 0.00 | ORB-short ORB[1065.00,1070.00] vol=3.2x ATR=3.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-27 10:00:00 | 1054.55 | 1061.66 | 0.00 | T1 1.5R @ 1054.55 |
| Target hit | 2024-02-27 13:20:00 | 1039.70 | 1038.89 | 0.00 | Trail-exit close>VWAP |

### Cycle 86 — BUY (started 2024-02-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-28 10:55:00 | 1055.60 | 1052.32 | 0.00 | ORB-long ORB[1033.95,1044.45] vol=2.1x ATR=3.76 |
| Stop hit — per-position SL triggered | 2024-02-28 11:10:00 | 1051.84 | 1052.52 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2024-02-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-29 11:15:00 | 1042.00 | 1045.39 | 0.00 | ORB-short ORB[1043.05,1053.30] vol=1.5x ATR=2.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-29 12:10:00 | 1037.73 | 1044.21 | 0.00 | T1 1.5R @ 1037.73 |
| Stop hit — per-position SL triggered | 2024-02-29 13:00:00 | 1042.00 | 1042.63 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2024-03-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-05 10:20:00 | 1082.00 | 1080.03 | 0.00 | ORB-long ORB[1068.10,1079.45] vol=2.1x ATR=3.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-05 13:05:00 | 1086.57 | 1082.30 | 0.00 | T1 1.5R @ 1086.57 |
| Target hit | 2024-03-05 15:20:00 | 1097.05 | 1092.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 89 — SELL (started 2024-03-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-07 09:40:00 | 1048.50 | 1052.17 | 0.00 | ORB-short ORB[1051.30,1058.65] vol=2.0x ATR=4.42 |
| Stop hit — per-position SL triggered | 2024-03-07 10:00:00 | 1052.92 | 1052.38 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2024-03-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-11 11:00:00 | 1056.05 | 1058.29 | 0.00 | ORB-short ORB[1060.50,1074.00] vol=2.1x ATR=3.13 |
| Stop hit — per-position SL triggered | 2024-03-11 11:05:00 | 1059.18 | 1058.28 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2024-04-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-18 09:45:00 | 1232.60 | 1235.92 | 0.00 | ORB-short ORB[1235.00,1243.90] vol=2.9x ATR=3.82 |
| Stop hit — per-position SL triggered | 2024-04-18 09:50:00 | 1236.42 | 1235.97 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2024-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 10:15:00 | 1326.90 | 1313.19 | 0.00 | ORB-long ORB[1304.00,1313.95] vol=2.1x ATR=6.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-24 11:25:00 | 1336.12 | 1324.40 | 0.00 | T1 1.5R @ 1336.12 |
| Target hit | 2024-04-24 15:20:00 | 1340.80 | 1336.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 93 — BUY (started 2024-04-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 09:55:00 | 1370.00 | 1359.15 | 0.00 | ORB-long ORB[1338.10,1351.25] vol=1.5x ATR=7.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-25 12:10:00 | 1381.28 | 1377.00 | 0.00 | T1 1.5R @ 1381.28 |
| Target hit | 2024-04-25 15:20:00 | 1417.00 | 1392.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 94 — BUY (started 2024-05-02 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-02 10:10:00 | 1458.00 | 1442.53 | 0.00 | ORB-long ORB[1425.95,1441.75] vol=1.5x ATR=7.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-02 11:55:00 | 1468.53 | 1456.15 | 0.00 | T1 1.5R @ 1468.53 |
| Target hit | 2024-05-02 14:35:00 | 1510.00 | 1511.42 | 0.00 | Trail-exit close<VWAP |

### Cycle 95 — SELL (started 2024-05-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 10:55:00 | 1478.15 | 1482.62 | 0.00 | ORB-short ORB[1480.55,1498.75] vol=1.6x ATR=4.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 11:30:00 | 1470.87 | 1477.41 | 0.00 | T1 1.5R @ 1470.87 |
| Stop hit — per-position SL triggered | 2024-05-09 11:35:00 | 1478.15 | 1475.44 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-16 10:45:00 | 1149.60 | 2023-05-16 10:50:00 | 1146.33 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-05-18 10:45:00 | 1167.20 | 2023-05-18 11:15:00 | 1162.80 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2023-05-18 10:45:00 | 1167.20 | 2023-05-18 12:25:00 | 1167.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-19 10:25:00 | 1183.05 | 2023-05-19 10:35:00 | 1178.24 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2023-05-23 11:05:00 | 1222.45 | 2023-05-23 11:45:00 | 1217.66 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-05-23 11:05:00 | 1222.45 | 2023-05-23 15:20:00 | 1196.00 | TARGET_HIT | 0.50 | 2.16% |
| SELL | retest1 | 2023-05-24 10:00:00 | 1179.00 | 2023-05-24 10:10:00 | 1172.46 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2023-05-24 10:00:00 | 1179.00 | 2023-05-24 10:30:00 | 1179.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-25 10:50:00 | 1180.60 | 2023-05-25 11:00:00 | 1176.50 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-05-31 10:45:00 | 1161.60 | 2023-05-31 11:00:00 | 1158.14 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-06-09 09:45:00 | 1192.05 | 2023-06-09 09:55:00 | 1195.49 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-06-12 09:35:00 | 1199.70 | 2023-06-12 15:10:00 | 1194.39 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2023-06-13 09:55:00 | 1199.85 | 2023-06-13 10:50:00 | 1196.24 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-06-14 10:50:00 | 1201.30 | 2023-06-14 11:00:00 | 1205.89 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-06-14 10:50:00 | 1201.30 | 2023-06-14 14:25:00 | 1202.65 | TARGET_HIT | 0.50 | 0.11% |
| SELL | retest1 | 2023-06-21 11:10:00 | 1204.15 | 2023-06-21 11:35:00 | 1207.32 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-06-22 10:00:00 | 1233.25 | 2023-06-22 10:05:00 | 1229.82 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-06-26 09:40:00 | 1205.90 | 2023-06-26 09:50:00 | 1201.58 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2023-06-28 10:25:00 | 1162.25 | 2023-06-28 11:10:00 | 1167.10 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2023-06-30 09:45:00 | 1191.15 | 2023-06-30 09:50:00 | 1195.98 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2023-07-04 09:40:00 | 1224.10 | 2023-07-04 09:45:00 | 1219.48 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2023-07-07 10:40:00 | 1210.45 | 2023-07-07 11:15:00 | 1206.38 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-07-10 10:15:00 | 1184.50 | 2023-07-10 10:30:00 | 1189.65 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2023-07-13 10:30:00 | 1193.30 | 2023-07-13 10:50:00 | 1187.02 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2023-07-13 10:30:00 | 1193.30 | 2023-07-13 12:20:00 | 1193.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-14 10:15:00 | 1176.55 | 2023-07-14 11:05:00 | 1180.61 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2023-07-17 10:50:00 | 1193.15 | 2023-07-17 13:05:00 | 1185.49 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2023-07-17 10:50:00 | 1193.15 | 2023-07-17 15:20:00 | 1177.05 | TARGET_HIT | 0.50 | 1.35% |
| BUY | retest1 | 2023-07-18 09:40:00 | 1192.05 | 2023-07-18 09:50:00 | 1200.21 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2023-07-18 09:40:00 | 1192.05 | 2023-07-18 10:30:00 | 1194.40 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2023-07-19 11:10:00 | 1218.05 | 2023-07-19 12:20:00 | 1213.21 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2023-07-21 10:40:00 | 1199.90 | 2023-07-21 11:45:00 | 1203.62 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-07-25 10:35:00 | 1190.20 | 2023-07-25 11:05:00 | 1183.60 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2023-07-25 10:35:00 | 1190.20 | 2023-07-25 14:50:00 | 1185.00 | TARGET_HIT | 0.50 | 0.44% |
| BUY | retest1 | 2023-07-27 10:20:00 | 1212.80 | 2023-07-27 11:00:00 | 1207.73 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2023-08-16 09:40:00 | 1049.80 | 2023-08-16 09:45:00 | 1053.72 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2023-08-17 09:35:00 | 1085.80 | 2023-08-17 09:40:00 | 1077.91 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2023-08-17 09:35:00 | 1085.80 | 2023-08-17 11:30:00 | 1072.30 | TARGET_HIT | 0.50 | 1.24% |
| BUY | retest1 | 2023-08-18 10:10:00 | 1091.00 | 2023-08-18 11:40:00 | 1099.75 | PARTIAL | 0.50 | 0.80% |
| BUY | retest1 | 2023-08-18 10:10:00 | 1091.00 | 2023-08-18 13:25:00 | 1091.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-23 10:15:00 | 1129.70 | 2023-08-23 10:20:00 | 1125.76 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2023-08-24 09:30:00 | 1132.05 | 2023-08-24 09:45:00 | 1137.04 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2023-08-25 09:40:00 | 1126.00 | 2023-08-25 09:50:00 | 1129.51 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-08-30 10:20:00 | 1142.85 | 2023-08-30 11:00:00 | 1139.88 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-09-01 09:55:00 | 1116.05 | 2023-09-01 10:55:00 | 1120.16 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-09-04 10:30:00 | 1147.65 | 2023-09-04 11:40:00 | 1143.33 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2023-09-08 10:40:00 | 1183.00 | 2023-09-08 10:50:00 | 1186.34 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-09-14 09:50:00 | 1206.25 | 2023-09-14 10:00:00 | 1201.04 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2023-09-15 09:45:00 | 1192.05 | 2023-09-15 09:55:00 | 1195.16 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-09-18 10:50:00 | 1180.00 | 2023-09-18 10:55:00 | 1174.05 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2023-09-18 10:50:00 | 1180.00 | 2023-09-18 11:05:00 | 1180.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-20 10:55:00 | 1171.05 | 2023-09-20 11:00:00 | 1165.57 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2023-09-20 10:55:00 | 1171.05 | 2023-09-20 12:40:00 | 1158.55 | TARGET_HIT | 0.50 | 1.07% |
| SELL | retest1 | 2023-09-21 09:45:00 | 1157.50 | 2023-09-21 09:50:00 | 1162.43 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2023-09-26 09:40:00 | 1184.30 | 2023-09-26 09:55:00 | 1178.20 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2023-09-26 09:40:00 | 1184.30 | 2023-09-26 14:35:00 | 1180.00 | TARGET_HIT | 0.50 | 0.36% |
| BUY | retest1 | 2023-09-28 09:40:00 | 1190.40 | 2023-09-28 12:35:00 | 1186.59 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-10-04 09:40:00 | 1172.95 | 2023-10-04 09:45:00 | 1169.26 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-10-05 10:20:00 | 1180.10 | 2023-10-05 11:40:00 | 1173.99 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2023-10-05 10:20:00 | 1180.10 | 2023-10-05 11:45:00 | 1180.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-06 10:55:00 | 1179.35 | 2023-10-06 11:25:00 | 1174.96 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2023-10-06 10:55:00 | 1179.35 | 2023-10-06 15:20:00 | 1167.85 | TARGET_HIT | 0.50 | 0.98% |
| SELL | retest1 | 2023-10-09 10:35:00 | 1143.35 | 2023-10-09 11:10:00 | 1137.87 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2023-10-09 10:35:00 | 1143.35 | 2023-10-09 14:35:00 | 1133.60 | TARGET_HIT | 0.50 | 0.85% |
| BUY | retest1 | 2023-10-11 09:55:00 | 1160.40 | 2023-10-11 10:05:00 | 1155.83 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2023-10-20 09:30:00 | 1135.10 | 2023-10-20 09:45:00 | 1138.95 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-11-08 11:05:00 | 1083.20 | 2023-11-08 11:15:00 | 1080.17 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-11-09 10:00:00 | 1057.05 | 2023-11-09 10:05:00 | 1052.35 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2023-11-09 10:00:00 | 1057.05 | 2023-11-09 10:15:00 | 1057.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-10 10:15:00 | 1078.55 | 2023-11-10 10:25:00 | 1075.32 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-11-13 10:10:00 | 1073.20 | 2023-11-13 10:15:00 | 1070.33 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-11-15 10:15:00 | 1090.70 | 2023-11-15 10:35:00 | 1085.99 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2023-11-20 10:10:00 | 1111.45 | 2023-11-20 10:25:00 | 1115.65 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2023-11-22 10:45:00 | 1089.15 | 2023-11-22 11:00:00 | 1091.86 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-11-24 10:55:00 | 1137.05 | 2023-11-24 11:30:00 | 1134.26 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-12-05 11:10:00 | 1186.15 | 2023-12-05 11:30:00 | 1189.65 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-12-07 10:20:00 | 1192.55 | 2023-12-07 10:30:00 | 1196.60 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2023-12-07 10:20:00 | 1192.55 | 2023-12-07 15:20:00 | 1211.20 | TARGET_HIT | 0.50 | 1.56% |
| SELL | retest1 | 2023-12-11 10:05:00 | 1194.05 | 2023-12-11 10:30:00 | 1189.42 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-12-11 10:05:00 | 1194.05 | 2023-12-11 10:50:00 | 1194.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-12 09:50:00 | 1206.35 | 2023-12-12 10:05:00 | 1210.28 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-12-12 09:50:00 | 1206.35 | 2023-12-12 10:15:00 | 1206.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-13 11:00:00 | 1207.30 | 2023-12-13 11:30:00 | 1202.00 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2023-12-13 11:00:00 | 1207.30 | 2023-12-13 15:20:00 | 1183.30 | TARGET_HIT | 0.50 | 1.99% |
| BUY | retest1 | 2023-12-15 09:35:00 | 1203.50 | 2023-12-15 09:40:00 | 1200.65 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-12-19 09:40:00 | 1204.50 | 2023-12-19 09:45:00 | 1200.56 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-12-20 11:00:00 | 1163.70 | 2023-12-20 11:10:00 | 1157.67 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2023-12-20 11:00:00 | 1163.70 | 2023-12-20 15:20:00 | 1113.00 | TARGET_HIT | 0.50 | 4.36% |
| SELL | retest1 | 2023-12-22 10:45:00 | 1109.80 | 2023-12-22 12:05:00 | 1104.21 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2023-12-22 10:45:00 | 1109.80 | 2023-12-22 15:20:00 | 1095.00 | TARGET_HIT | 0.50 | 1.33% |
| SELL | retest1 | 2023-12-26 10:50:00 | 1088.10 | 2023-12-26 11:20:00 | 1091.35 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-12-27 09:30:00 | 1115.35 | 2023-12-27 09:45:00 | 1111.41 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-01-01 10:40:00 | 1113.20 | 2024-01-01 10:50:00 | 1116.42 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-01-04 10:55:00 | 1122.30 | 2024-01-04 11:00:00 | 1125.04 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-01-05 10:45:00 | 1125.95 | 2024-01-05 11:00:00 | 1129.58 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-01-05 10:45:00 | 1125.95 | 2024-01-05 14:20:00 | 1135.45 | TARGET_HIT | 0.50 | 0.84% |
| BUY | retest1 | 2024-01-11 10:15:00 | 1156.95 | 2024-01-11 10:25:00 | 1162.58 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-01-11 10:15:00 | 1156.95 | 2024-01-11 10:30:00 | 1156.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-12 11:15:00 | 1165.50 | 2024-01-12 11:50:00 | 1169.94 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-01-12 11:15:00 | 1165.50 | 2024-01-12 14:10:00 | 1165.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-17 11:15:00 | 1141.65 | 2024-01-17 12:45:00 | 1145.26 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-01-18 09:35:00 | 1132.70 | 2024-01-18 09:45:00 | 1128.07 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-01-18 09:35:00 | 1132.70 | 2024-01-18 11:45:00 | 1129.15 | TARGET_HIT | 0.50 | 0.31% |
| BUY | retest1 | 2024-01-30 11:00:00 | 1143.95 | 2024-01-30 11:05:00 | 1141.13 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-01-31 09:45:00 | 1131.90 | 2024-01-31 09:55:00 | 1139.04 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-01-31 09:45:00 | 1131.90 | 2024-01-31 13:05:00 | 1133.25 | TARGET_HIT | 0.50 | 0.12% |
| BUY | retest1 | 2024-02-02 10:00:00 | 1137.15 | 2024-02-02 10:10:00 | 1133.69 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-02-05 10:35:00 | 1103.75 | 2024-02-05 11:10:00 | 1097.91 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-02-05 10:35:00 | 1103.75 | 2024-02-05 15:20:00 | 1075.00 | TARGET_HIT | 0.50 | 2.60% |
| BUY | retest1 | 2024-02-06 10:50:00 | 1109.40 | 2024-02-06 12:10:00 | 1115.34 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-02-06 10:50:00 | 1109.40 | 2024-02-06 15:20:00 | 1119.90 | TARGET_HIT | 0.50 | 0.95% |
| SELL | retest1 | 2024-02-09 11:05:00 | 1120.00 | 2024-02-09 12:40:00 | 1123.36 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-02-16 10:45:00 | 1120.75 | 2024-02-16 10:55:00 | 1116.44 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-02-26 09:50:00 | 1079.30 | 2024-02-26 10:00:00 | 1074.59 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-02-26 09:50:00 | 1079.30 | 2024-02-26 10:20:00 | 1079.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-27 09:45:00 | 1059.15 | 2024-02-27 10:00:00 | 1054.55 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-02-27 09:45:00 | 1059.15 | 2024-02-27 13:20:00 | 1039.70 | TARGET_HIT | 0.50 | 1.84% |
| BUY | retest1 | 2024-02-28 10:55:00 | 1055.60 | 2024-02-28 11:10:00 | 1051.84 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-02-29 11:15:00 | 1042.00 | 2024-02-29 12:10:00 | 1037.73 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-02-29 11:15:00 | 1042.00 | 2024-02-29 13:00:00 | 1042.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-05 10:20:00 | 1082.00 | 2024-03-05 13:05:00 | 1086.57 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-03-05 10:20:00 | 1082.00 | 2024-03-05 15:20:00 | 1097.05 | TARGET_HIT | 0.50 | 1.39% |
| SELL | retest1 | 2024-03-07 09:40:00 | 1048.50 | 2024-03-07 10:00:00 | 1052.92 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-03-11 11:00:00 | 1056.05 | 2024-03-11 11:05:00 | 1059.18 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-04-18 09:45:00 | 1232.60 | 2024-04-18 09:50:00 | 1236.42 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-04-24 10:15:00 | 1326.90 | 2024-04-24 11:25:00 | 1336.12 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-04-24 10:15:00 | 1326.90 | 2024-04-24 15:20:00 | 1340.80 | TARGET_HIT | 0.50 | 1.05% |
| BUY | retest1 | 2024-04-25 09:55:00 | 1370.00 | 2024-04-25 12:10:00 | 1381.28 | PARTIAL | 0.50 | 0.82% |
| BUY | retest1 | 2024-04-25 09:55:00 | 1370.00 | 2024-04-25 15:20:00 | 1417.00 | TARGET_HIT | 0.50 | 3.43% |
| BUY | retest1 | 2024-05-02 10:10:00 | 1458.00 | 2024-05-02 11:55:00 | 1468.53 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2024-05-02 10:10:00 | 1458.00 | 2024-05-02 14:35:00 | 1510.00 | TARGET_HIT | 0.50 | 3.57% |
| SELL | retest1 | 2024-05-09 10:55:00 | 1478.15 | 2024-05-09 11:30:00 | 1470.87 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-05-09 10:55:00 | 1478.15 | 2024-05-09 11:35:00 | 1478.15 | STOP_HIT | 0.50 | 0.00% |
