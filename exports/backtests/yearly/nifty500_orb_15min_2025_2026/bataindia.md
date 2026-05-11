# Bata India Ltd. (BATAINDIA)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 722.80
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
| ENTRY1 | 105 |
| ENTRY2 | 0 |
| PARTIAL | 35 |
| TARGET_HIT | 20 |
| STOP_HIT | 85 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 140 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 55 / 85
- **Target hits / Stop hits / Partials:** 20 / 85 / 35
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 14.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 54 | 14 | 25.9% | 5 | 40 | 9 | 0.01% | 0.5% |
| BUY @ 2nd Alert (retest1) | 54 | 14 | 25.9% | 5 | 40 | 9 | 0.01% | 0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 86 | 41 | 47.7% | 15 | 45 | 26 | 0.17% | 14.2% |
| SELL @ 2nd Alert (retest1) | 86 | 41 | 47.7% | 15 | 45 | 26 | 0.17% | 14.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 140 | 55 | 39.3% | 20 | 85 | 35 | 0.10% | 14.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-12 10:25:00 | 1213.50 | 1214.73 | 0.00 | ORB-short ORB[1214.40,1227.00] vol=3.6x ATR=6.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-12 12:00:00 | 1203.65 | 1211.27 | 0.00 | T1 1.5R @ 1203.65 |
| Target hit | 2025-05-12 12:40:00 | 1211.00 | 1210.85 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — BUY (started 2025-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 10:15:00 | 1226.80 | 1218.17 | 0.00 | ORB-long ORB[1206.20,1221.00] vol=4.4x ATR=4.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-13 10:20:00 | 1233.31 | 1219.96 | 0.00 | T1 1.5R @ 1233.31 |
| Stop hit — per-position SL triggered | 2025-05-13 10:25:00 | 1226.80 | 1220.25 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 10:15:00 | 1218.10 | 1213.83 | 0.00 | ORB-long ORB[1206.50,1214.00] vol=2.5x ATR=4.22 |
| Stop hit — per-position SL triggered | 2025-05-15 10:35:00 | 1213.88 | 1214.16 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-19 09:35:00 | 1244.50 | 1252.40 | 0.00 | ORB-short ORB[1250.00,1259.00] vol=1.8x ATR=4.12 |
| Stop hit — per-position SL triggered | 2025-05-19 10:10:00 | 1248.62 | 1250.71 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 09:45:00 | 1233.50 | 1223.03 | 0.00 | ORB-long ORB[1212.80,1223.10] vol=1.7x ATR=4.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-21 10:05:00 | 1239.57 | 1228.73 | 0.00 | T1 1.5R @ 1239.57 |
| Target hit | 2025-05-21 15:20:00 | 1257.70 | 1250.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2025-05-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 10:25:00 | 1275.90 | 1265.50 | 0.00 | ORB-long ORB[1257.50,1267.00] vol=3.2x ATR=4.17 |
| Stop hit — per-position SL triggered | 2025-05-23 10:30:00 | 1271.73 | 1265.85 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-05-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 09:55:00 | 1292.60 | 1285.70 | 0.00 | ORB-long ORB[1273.30,1288.00] vol=5.2x ATR=4.91 |
| Stop hit — per-position SL triggered | 2025-05-26 10:10:00 | 1287.69 | 1287.13 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-30 11:15:00 | 1264.90 | 1258.74 | 0.00 | ORB-long ORB[1254.00,1262.40] vol=2.0x ATR=3.04 |
| Stop hit — per-position SL triggered | 2025-05-30 13:40:00 | 1261.86 | 1262.84 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 11:00:00 | 1269.10 | 1262.70 | 0.00 | ORB-long ORB[1251.50,1266.30] vol=2.4x ATR=2.59 |
| Stop hit — per-position SL triggered | 2025-06-02 11:10:00 | 1266.51 | 1263.58 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 11:15:00 | 1226.10 | 1229.57 | 0.00 | ORB-short ORB[1228.80,1237.50] vol=1.8x ATR=2.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-04 13:40:00 | 1222.48 | 1227.48 | 0.00 | T1 1.5R @ 1222.48 |
| Target hit | 2025-06-04 15:20:00 | 1222.40 | 1225.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2025-06-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-09 09:45:00 | 1219.60 | 1222.51 | 0.00 | ORB-short ORB[1220.00,1230.60] vol=1.7x ATR=2.77 |
| Target hit | 2025-06-09 15:20:00 | 1218.00 | 1220.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2025-06-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-11 11:15:00 | 1214.90 | 1218.94 | 0.00 | ORB-short ORB[1218.00,1226.00] vol=5.1x ATR=1.60 |
| Stop hit — per-position SL triggered | 2025-06-11 12:35:00 | 1216.50 | 1217.48 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-12 10:25:00 | 1225.10 | 1222.16 | 0.00 | ORB-long ORB[1217.00,1225.00] vol=1.8x ATR=2.23 |
| Stop hit — per-position SL triggered | 2025-06-12 10:35:00 | 1222.87 | 1222.26 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 10:15:00 | 1223.60 | 1219.25 | 0.00 | ORB-long ORB[1212.30,1219.10] vol=2.0x ATR=2.36 |
| Stop hit — per-position SL triggered | 2025-06-17 10:45:00 | 1221.24 | 1220.34 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-06-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-18 10:10:00 | 1221.30 | 1219.32 | 0.00 | ORB-long ORB[1214.90,1221.10] vol=1.5x ATR=1.97 |
| Stop hit — per-position SL triggered | 2025-06-18 10:20:00 | 1219.33 | 1219.34 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-06-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 09:35:00 | 1211.80 | 1209.30 | 0.00 | ORB-long ORB[1203.20,1211.50] vol=1.6x ATR=2.76 |
| Stop hit — per-position SL triggered | 2025-06-25 09:50:00 | 1209.04 | 1209.44 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-06-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 10:55:00 | 1209.00 | 1213.76 | 0.00 | ORB-short ORB[1214.00,1219.80] vol=2.0x ATR=1.92 |
| Stop hit — per-position SL triggered | 2025-06-26 11:15:00 | 1210.92 | 1213.44 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 11:10:00 | 1224.90 | 1228.51 | 0.00 | ORB-short ORB[1226.90,1235.00] vol=1.6x ATR=1.92 |
| Stop hit — per-position SL triggered | 2025-07-02 11:30:00 | 1226.82 | 1228.46 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 09:45:00 | 1234.10 | 1230.32 | 0.00 | ORB-long ORB[1227.00,1234.00] vol=1.8x ATR=2.60 |
| Stop hit — per-position SL triggered | 2025-07-03 09:50:00 | 1231.50 | 1230.61 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-07 09:35:00 | 1233.30 | 1236.14 | 0.00 | ORB-short ORB[1234.40,1240.00] vol=1.7x ATR=3.11 |
| Stop hit — per-position SL triggered | 2025-07-07 09:45:00 | 1236.41 | 1236.99 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-07-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 10:35:00 | 1262.50 | 1253.16 | 0.00 | ORB-long ORB[1238.40,1250.90] vol=2.2x ATR=4.08 |
| Stop hit — per-position SL triggered | 2025-07-09 11:45:00 | 1258.42 | 1255.21 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-07-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 10:00:00 | 1281.20 | 1273.79 | 0.00 | ORB-long ORB[1257.90,1269.50] vol=5.5x ATR=4.60 |
| Stop hit — per-position SL triggered | 2025-07-10 10:10:00 | 1276.60 | 1275.02 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-07-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 09:30:00 | 1254.20 | 1257.09 | 0.00 | ORB-short ORB[1257.50,1263.90] vol=2.1x ATR=4.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 10:50:00 | 1247.73 | 1254.52 | 0.00 | T1 1.5R @ 1247.73 |
| Target hit | 2025-07-11 15:20:00 | 1237.50 | 1245.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — BUY (started 2025-07-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 11:00:00 | 1267.30 | 1253.79 | 0.00 | ORB-long ORB[1240.90,1255.60] vol=7.8x ATR=3.93 |
| Stop hit — per-position SL triggered | 2025-07-15 11:10:00 | 1263.37 | 1258.69 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-07-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 11:00:00 | 1250.10 | 1261.29 | 0.00 | ORB-short ORB[1258.00,1273.10] vol=6.1x ATR=2.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-16 12:15:00 | 1246.07 | 1256.81 | 0.00 | T1 1.5R @ 1246.07 |
| Target hit | 2025-07-16 15:20:00 | 1243.90 | 1250.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — SELL (started 2025-07-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 09:45:00 | 1236.90 | 1240.77 | 0.00 | ORB-short ORB[1240.90,1248.40] vol=1.5x ATR=2.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 14:25:00 | 1233.42 | 1236.20 | 0.00 | T1 1.5R @ 1233.42 |
| Target hit | 2025-07-17 15:20:00 | 1229.40 | 1233.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — SELL (started 2025-07-18 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:20:00 | 1219.70 | 1226.05 | 0.00 | ORB-short ORB[1229.90,1235.40] vol=3.2x ATR=2.54 |
| Stop hit — per-position SL triggered | 2025-07-18 10:55:00 | 1222.24 | 1224.39 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-07-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 10:55:00 | 1206.50 | 1208.92 | 0.00 | ORB-short ORB[1206.80,1214.80] vol=2.3x ATR=1.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-23 11:30:00 | 1203.59 | 1207.84 | 0.00 | T1 1.5R @ 1203.59 |
| Stop hit — per-position SL triggered | 2025-07-23 11:45:00 | 1206.50 | 1207.74 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-07-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 10:00:00 | 1213.60 | 1211.33 | 0.00 | ORB-long ORB[1206.00,1213.10] vol=2.2x ATR=2.57 |
| Stop hit — per-position SL triggered | 2025-07-24 10:05:00 | 1211.03 | 1211.50 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-07-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-29 09:55:00 | 1189.00 | 1194.89 | 0.00 | ORB-short ORB[1194.00,1203.20] vol=6.1x ATR=3.16 |
| Stop hit — per-position SL triggered | 2025-07-29 10:00:00 | 1192.16 | 1193.77 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 11:15:00 | 1182.50 | 1185.72 | 0.00 | ORB-short ORB[1182.70,1190.10] vol=2.0x ATR=1.64 |
| Stop hit — per-position SL triggered | 2025-08-05 11:50:00 | 1184.14 | 1185.62 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-08-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-11 10:40:00 | 1189.30 | 1182.35 | 0.00 | ORB-long ORB[1178.10,1186.10] vol=2.0x ATR=2.65 |
| Stop hit — per-position SL triggered | 2025-08-11 10:50:00 | 1186.65 | 1183.08 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-08-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-13 09:30:00 | 1119.00 | 1123.24 | 0.00 | ORB-short ORB[1119.10,1135.00] vol=1.6x ATR=4.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 10:30:00 | 1112.35 | 1120.17 | 0.00 | T1 1.5R @ 1112.35 |
| Target hit | 2025-08-13 15:20:00 | 1091.90 | 1109.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — SELL (started 2025-08-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 09:55:00 | 1083.90 | 1089.18 | 0.00 | ORB-short ORB[1089.50,1098.90] vol=2.2x ATR=3.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-14 10:15:00 | 1078.70 | 1086.11 | 0.00 | T1 1.5R @ 1078.70 |
| Stop hit — per-position SL triggered | 2025-08-14 11:05:00 | 1083.90 | 1083.92 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 09:30:00 | 1106.20 | 1112.46 | 0.00 | ORB-short ORB[1109.00,1122.90] vol=2.3x ATR=3.54 |
| Stop hit — per-position SL triggered | 2025-08-22 10:35:00 | 1109.74 | 1110.69 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-25 10:15:00 | 1096.20 | 1103.35 | 0.00 | ORB-short ORB[1101.40,1115.00] vol=1.8x ATR=2.83 |
| Stop hit — per-position SL triggered | 2025-08-25 11:45:00 | 1099.03 | 1100.90 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-08-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:45:00 | 1084.80 | 1088.12 | 0.00 | ORB-short ORB[1088.10,1097.10] vol=1.9x ATR=2.86 |
| Stop hit — per-position SL triggered | 2025-08-26 11:10:00 | 1087.66 | 1087.37 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-08-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-28 09:35:00 | 1076.90 | 1070.32 | 0.00 | ORB-long ORB[1065.00,1070.90] vol=2.1x ATR=3.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 09:45:00 | 1082.89 | 1074.73 | 0.00 | T1 1.5R @ 1082.89 |
| Stop hit — per-position SL triggered | 2025-08-28 10:10:00 | 1076.90 | 1076.99 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-09-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 10:50:00 | 1111.00 | 1104.17 | 0.00 | ORB-long ORB[1099.70,1110.30] vol=2.8x ATR=4.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 11:00:00 | 1117.27 | 1106.22 | 0.00 | T1 1.5R @ 1117.27 |
| Target hit | 2025-09-01 15:20:00 | 1115.90 | 1111.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — BUY (started 2025-09-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 10:00:00 | 1134.40 | 1126.56 | 0.00 | ORB-long ORB[1117.10,1132.00] vol=2.8x ATR=3.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 10:10:00 | 1140.31 | 1131.64 | 0.00 | T1 1.5R @ 1140.31 |
| Target hit | 2025-09-02 14:15:00 | 1141.50 | 1142.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 41 — SELL (started 2025-09-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-11 10:20:00 | 1231.20 | 1235.53 | 0.00 | ORB-short ORB[1235.00,1243.90] vol=4.8x ATR=3.44 |
| Stop hit — per-position SL triggered | 2025-09-11 10:25:00 | 1234.64 | 1235.37 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-09-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 10:00:00 | 1256.10 | 1247.61 | 0.00 | ORB-long ORB[1235.00,1248.70] vol=7.3x ATR=4.41 |
| Stop hit — per-position SL triggered | 2025-09-12 10:05:00 | 1251.69 | 1248.66 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-09-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 10:05:00 | 1277.00 | 1273.57 | 0.00 | ORB-long ORB[1261.70,1274.20] vol=11.7x ATR=3.85 |
| Stop hit — per-position SL triggered | 2025-09-16 10:15:00 | 1273.15 | 1273.67 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-09-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 10:55:00 | 1264.20 | 1270.37 | 0.00 | ORB-short ORB[1270.50,1280.90] vol=1.9x ATR=2.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 13:25:00 | 1260.19 | 1267.83 | 0.00 | T1 1.5R @ 1260.19 |
| Target hit | 2025-09-17 15:20:00 | 1259.80 | 1264.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — SELL (started 2025-09-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 09:55:00 | 1237.60 | 1245.41 | 0.00 | ORB-short ORB[1241.10,1249.50] vol=4.7x ATR=3.71 |
| Stop hit — per-position SL triggered | 2025-09-19 10:00:00 | 1241.31 | 1245.10 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-09-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-22 10:30:00 | 1224.30 | 1234.68 | 0.00 | ORB-short ORB[1235.50,1249.50] vol=1.5x ATR=3.51 |
| Stop hit — per-position SL triggered | 2025-09-22 11:15:00 | 1227.81 | 1233.10 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-09-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-25 09:40:00 | 1205.20 | 1209.71 | 0.00 | ORB-short ORB[1205.60,1218.10] vol=2.9x ATR=3.46 |
| Stop hit — per-position SL triggered | 2025-09-25 09:50:00 | 1208.66 | 1209.08 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-10-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 09:35:00 | 1189.20 | 1185.01 | 0.00 | ORB-long ORB[1164.50,1176.00] vol=13.5x ATR=5.26 |
| Stop hit — per-position SL triggered | 2025-10-01 09:45:00 | 1183.94 | 1185.06 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-10-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 09:50:00 | 1210.50 | 1207.25 | 0.00 | ORB-long ORB[1188.30,1199.00] vol=12.0x ATR=4.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 10:00:00 | 1216.98 | 1208.20 | 0.00 | T1 1.5R @ 1216.98 |
| Stop hit — per-position SL triggered | 2025-10-03 10:15:00 | 1210.50 | 1208.37 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-10-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 10:30:00 | 1215.20 | 1210.02 | 0.00 | ORB-long ORB[1201.00,1214.50] vol=4.2x ATR=5.20 |
| Stop hit — per-position SL triggered | 2025-10-06 10:45:00 | 1210.00 | 1210.08 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-10-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 09:50:00 | 1231.60 | 1220.84 | 0.00 | ORB-long ORB[1207.00,1222.00] vol=3.5x ATR=5.11 |
| Stop hit — per-position SL triggered | 2025-10-07 10:05:00 | 1226.49 | 1223.29 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-10-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 09:55:00 | 1211.80 | 1215.07 | 0.00 | ORB-short ORB[1214.10,1222.00] vol=2.9x ATR=3.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 10:10:00 | 1207.22 | 1214.06 | 0.00 | T1 1.5R @ 1207.22 |
| Stop hit — per-position SL triggered | 2025-10-08 10:30:00 | 1211.80 | 1213.40 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-10-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:35:00 | 1181.80 | 1179.18 | 0.00 | ORB-long ORB[1171.60,1180.50] vol=1.9x ATR=2.82 |
| Stop hit — per-position SL triggered | 2025-10-10 09:55:00 | 1178.98 | 1180.03 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-10-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 11:10:00 | 1158.90 | 1165.89 | 0.00 | ORB-short ORB[1162.50,1175.50] vol=3.2x ATR=2.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 11:45:00 | 1155.48 | 1163.76 | 0.00 | T1 1.5R @ 1155.48 |
| Target hit | 2025-10-13 15:20:00 | 1149.00 | 1154.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — SELL (started 2025-10-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 09:35:00 | 1147.70 | 1152.29 | 0.00 | ORB-short ORB[1150.50,1157.80] vol=1.9x ATR=2.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 09:50:00 | 1143.89 | 1150.09 | 0.00 | T1 1.5R @ 1143.89 |
| Target hit | 2025-10-14 15:20:00 | 1125.20 | 1133.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — SELL (started 2025-10-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-15 09:50:00 | 1112.00 | 1116.38 | 0.00 | ORB-short ORB[1113.30,1128.00] vol=2.5x ATR=2.83 |
| Stop hit — per-position SL triggered | 2025-10-15 10:15:00 | 1114.83 | 1114.86 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-10-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 09:40:00 | 1156.00 | 1151.91 | 0.00 | ORB-long ORB[1144.30,1155.00] vol=2.6x ATR=4.32 |
| Stop hit — per-position SL triggered | 2025-10-20 09:45:00 | 1151.68 | 1152.01 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-10-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 10:30:00 | 1175.20 | 1172.11 | 0.00 | ORB-long ORB[1161.90,1174.80] vol=2.1x ATR=2.91 |
| Stop hit — per-position SL triggered | 2025-10-23 11:10:00 | 1172.29 | 1172.34 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-10-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-30 10:25:00 | 1080.60 | 1077.81 | 0.00 | ORB-long ORB[1069.90,1080.00] vol=7.0x ATR=3.04 |
| Stop hit — per-position SL triggered | 2025-10-30 11:25:00 | 1077.56 | 1078.91 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-11-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 09:35:00 | 1052.60 | 1059.54 | 0.00 | ORB-short ORB[1058.80,1066.00] vol=1.9x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 10:15:00 | 1048.95 | 1054.12 | 0.00 | T1 1.5R @ 1048.95 |
| Stop hit — per-position SL triggered | 2025-11-06 12:00:00 | 1052.60 | 1053.14 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-11-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-10 10:55:00 | 1048.50 | 1052.48 | 0.00 | ORB-short ORB[1049.80,1065.00] vol=1.8x ATR=2.68 |
| Stop hit — per-position SL triggered | 2025-11-10 15:20:00 | 1049.60 | 1050.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — SELL (started 2025-11-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-12 10:00:00 | 1045.70 | 1048.21 | 0.00 | ORB-short ORB[1047.30,1053.20] vol=2.8x ATR=1.85 |
| Stop hit — per-position SL triggered | 2025-11-12 10:05:00 | 1047.55 | 1048.08 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-11-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 09:50:00 | 1006.00 | 1008.04 | 0.00 | ORB-short ORB[1007.40,1012.40] vol=2.5x ATR=1.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 09:55:00 | 1003.23 | 1006.27 | 0.00 | T1 1.5R @ 1003.23 |
| Target hit | 2025-11-21 15:05:00 | 1001.00 | 1000.99 | 0.00 | Trail-exit close>VWAP |

### Cycle 64 — BUY (started 2025-11-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-24 10:55:00 | 1009.80 | 1002.31 | 0.00 | ORB-long ORB[998.70,1008.00] vol=2.0x ATR=2.25 |
| Stop hit — per-position SL triggered | 2025-11-24 11:05:00 | 1007.55 | 1002.54 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-12-01 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 09:50:00 | 991.25 | 995.66 | 0.00 | ORB-short ORB[994.00,1004.05] vol=2.3x ATR=1.64 |
| Stop hit — per-position SL triggered | 2025-12-01 10:00:00 | 992.89 | 994.77 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-12-03 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-03 10:25:00 | 968.55 | 965.72 | 0.00 | ORB-long ORB[959.50,966.50] vol=1.6x ATR=2.50 |
| Stop hit — per-position SL triggered | 2025-12-03 11:05:00 | 966.05 | 965.91 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-12-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 10:50:00 | 963.20 | 958.03 | 0.00 | ORB-long ORB[953.05,960.30] vol=1.8x ATR=1.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 11:25:00 | 966.14 | 958.91 | 0.00 | T1 1.5R @ 966.14 |
| Stop hit — per-position SL triggered | 2025-12-05 12:00:00 | 963.20 | 959.71 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-12-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 11:10:00 | 954.00 | 959.78 | 0.00 | ORB-short ORB[960.00,972.00] vol=1.6x ATR=1.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 11:30:00 | 951.61 | 958.94 | 0.00 | T1 1.5R @ 951.61 |
| Stop hit — per-position SL triggered | 2025-12-08 11:40:00 | 954.00 | 958.69 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-12-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 10:05:00 | 953.10 | 946.53 | 0.00 | ORB-long ORB[942.00,951.20] vol=1.6x ATR=3.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 10:20:00 | 957.86 | 948.64 | 0.00 | T1 1.5R @ 957.86 |
| Target hit | 2025-12-09 15:20:00 | 983.50 | 975.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 70 — SELL (started 2025-12-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-22 11:05:00 | 947.40 | 949.80 | 0.00 | ORB-short ORB[948.00,956.70] vol=2.3x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-22 11:10:00 | 945.09 | 948.99 | 0.00 | T1 1.5R @ 945.09 |
| Stop hit — per-position SL triggered | 2025-12-22 11:20:00 | 947.40 | 948.77 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-12-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 09:55:00 | 939.05 | 941.88 | 0.00 | ORB-short ORB[940.80,950.20] vol=3.6x ATR=1.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 10:15:00 | 936.22 | 940.89 | 0.00 | T1 1.5R @ 936.22 |
| Stop hit — per-position SL triggered | 2025-12-29 10:40:00 | 939.05 | 940.41 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-12-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 09:45:00 | 942.35 | 940.88 | 0.00 | ORB-long ORB[936.00,942.05] vol=2.1x ATR=1.58 |
| Stop hit — per-position SL triggered | 2025-12-31 09:50:00 | 940.77 | 941.09 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2026-01-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 09:30:00 | 953.00 | 954.77 | 0.00 | ORB-short ORB[953.85,958.60] vol=1.7x ATR=1.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 09:40:00 | 950.28 | 953.87 | 0.00 | T1 1.5R @ 950.28 |
| Stop hit — per-position SL triggered | 2026-01-06 09:45:00 | 953.00 | 953.83 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-01-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 09:40:00 | 933.00 | 935.44 | 0.00 | ORB-short ORB[934.00,938.90] vol=1.6x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 09:45:00 | 930.77 | 932.93 | 0.00 | T1 1.5R @ 930.77 |
| Target hit | 2026-01-08 14:55:00 | 928.05 | 927.61 | 0.00 | Trail-exit close>VWAP |

### Cycle 75 — SELL (started 2026-01-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-09 09:40:00 | 916.45 | 919.19 | 0.00 | ORB-short ORB[916.60,925.00] vol=1.7x ATR=2.23 |
| Stop hit — per-position SL triggered | 2026-01-09 09:45:00 | 918.68 | 919.18 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2026-01-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-12 10:35:00 | 909.30 | 913.78 | 0.00 | ORB-short ORB[911.10,923.40] vol=2.0x ATR=2.42 |
| Stop hit — per-position SL triggered | 2026-01-12 15:20:00 | 909.85 | 910.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 77 — SELL (started 2026-01-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 10:45:00 | 907.05 | 909.39 | 0.00 | ORB-short ORB[908.00,913.80] vol=3.7x ATR=1.85 |
| Target hit | 2026-01-14 15:20:00 | 905.35 | 907.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 78 — SELL (started 2026-01-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-16 09:35:00 | 900.00 | 902.02 | 0.00 | ORB-short ORB[900.60,909.65] vol=1.6x ATR=1.48 |
| Stop hit — per-position SL triggered | 2026-01-16 09:45:00 | 901.48 | 901.98 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-01-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-19 09:40:00 | 891.00 | 894.79 | 0.00 | ORB-short ORB[892.40,902.00] vol=2.5x ATR=2.28 |
| Stop hit — per-position SL triggered | 2026-01-19 10:10:00 | 893.28 | 894.18 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2026-01-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-20 10:45:00 | 902.65 | 894.82 | 0.00 | ORB-long ORB[883.05,893.00] vol=10.9x ATR=3.96 |
| Stop hit — per-position SL triggered | 2026-01-20 10:50:00 | 898.69 | 895.38 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2026-01-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-21 10:40:00 | 874.00 | 880.54 | 0.00 | ORB-short ORB[877.55,888.25] vol=1.9x ATR=3.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 11:25:00 | 868.90 | 878.02 | 0.00 | T1 1.5R @ 868.90 |
| Stop hit — per-position SL triggered | 2026-01-21 11:55:00 | 874.00 | 877.52 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2026-01-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-23 09:50:00 | 871.20 | 876.02 | 0.00 | ORB-short ORB[875.05,884.90] vol=1.6x ATR=2.82 |
| Stop hit — per-position SL triggered | 2026-01-23 10:20:00 | 874.02 | 875.29 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2026-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 09:30:00 | 861.65 | 854.95 | 0.00 | ORB-long ORB[847.25,854.00] vol=3.6x ATR=2.77 |
| Stop hit — per-position SL triggered | 2026-01-30 09:45:00 | 858.88 | 857.19 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2026-02-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 10:55:00 | 852.75 | 855.75 | 0.00 | ORB-short ORB[855.25,863.25] vol=5.9x ATR=1.49 |
| Stop hit — per-position SL triggered | 2026-02-05 12:30:00 | 854.24 | 855.10 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2026-02-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 11:00:00 | 844.20 | 846.23 | 0.00 | ORB-short ORB[844.40,852.45] vol=2.9x ATR=2.06 |
| Stop hit — per-position SL triggered | 2026-02-06 11:20:00 | 846.26 | 846.13 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2026-02-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 09:30:00 | 862.00 | 857.76 | 0.00 | ORB-long ORB[851.10,859.10] vol=2.4x ATR=1.87 |
| Stop hit — per-position SL triggered | 2026-02-09 09:35:00 | 860.13 | 858.33 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 11:15:00 | 866.05 | 870.39 | 0.00 | ORB-short ORB[867.90,880.70] vol=7.3x ATR=2.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 11:20:00 | 862.00 | 869.78 | 0.00 | T1 1.5R @ 862.00 |
| Stop hit — per-position SL triggered | 2026-02-13 12:05:00 | 866.05 | 868.89 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2026-02-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 11:05:00 | 841.00 | 846.40 | 0.00 | ORB-short ORB[842.45,851.90] vol=8.2x ATR=2.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:15:00 | 836.84 | 845.38 | 0.00 | T1 1.5R @ 836.84 |
| Target hit | 2026-02-16 15:20:00 | 828.95 | 834.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 89 — BUY (started 2026-02-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:35:00 | 837.25 | 828.47 | 0.00 | ORB-long ORB[823.75,829.00] vol=1.5x ATR=2.81 |
| Stop hit — per-position SL triggered | 2026-02-17 09:40:00 | 834.44 | 829.79 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:15:00 | 832.60 | 834.67 | 0.00 | ORB-short ORB[833.90,839.90] vol=4.8x ATR=1.72 |
| Stop hit — per-position SL triggered | 2026-02-19 11:25:00 | 834.32 | 834.62 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 801.30 | 805.84 | 0.00 | ORB-short ORB[803.80,815.00] vol=2.3x ATR=2.94 |
| Stop hit — per-position SL triggered | 2026-02-24 10:05:00 | 804.24 | 805.12 | 0.00 | SL hit |

### Cycle 92 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:15:00 | 793.25 | 796.35 | 0.00 | ORB-short ORB[794.15,800.45] vol=2.5x ATR=1.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:20:00 | 790.37 | 795.12 | 0.00 | T1 1.5R @ 790.37 |
| Stop hit — per-position SL triggered | 2026-02-27 12:25:00 | 793.25 | 792.44 | 0.00 | SL hit |

### Cycle 93 — SELL (started 2026-03-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:30:00 | 753.85 | 757.24 | 0.00 | ORB-short ORB[755.00,762.55] vol=4.2x ATR=3.34 |
| Stop hit — per-position SL triggered | 2026-03-04 10:00:00 | 757.19 | 756.37 | 0.00 | SL hit |

### Cycle 94 — BUY (started 2026-03-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 09:30:00 | 748.50 | 745.10 | 0.00 | ORB-long ORB[738.80,746.80] vol=2.6x ATR=3.18 |
| Stop hit — per-position SL triggered | 2026-03-06 10:00:00 | 745.32 | 745.98 | 0.00 | SL hit |

### Cycle 95 — SELL (started 2026-03-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:50:00 | 705.85 | 708.05 | 0.00 | ORB-short ORB[707.60,713.60] vol=2.5x ATR=2.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:10:00 | 702.50 | 706.93 | 0.00 | T1 1.5R @ 702.50 |
| Target hit | 2026-03-13 11:50:00 | 703.30 | 702.17 | 0.00 | Trail-exit close>VWAP |

### Cycle 96 — BUY (started 2026-03-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 11:00:00 | 683.10 | 678.34 | 0.00 | ORB-long ORB[672.90,683.00] vol=3.2x ATR=2.92 |
| Stop hit — per-position SL triggered | 2026-03-18 11:05:00 | 680.18 | 678.58 | 0.00 | SL hit |

### Cycle 97 — SELL (started 2026-04-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:30:00 | 697.50 | 703.15 | 0.00 | ORB-short ORB[701.55,709.00] vol=2.3x ATR=2.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 09:40:00 | 693.54 | 701.81 | 0.00 | T1 1.5R @ 693.54 |
| Stop hit — per-position SL triggered | 2026-04-09 10:40:00 | 697.50 | 698.01 | 0.00 | SL hit |

### Cycle 98 — BUY (started 2026-04-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:50:00 | 713.45 | 711.22 | 0.00 | ORB-long ORB[702.00,709.85] vol=1.7x ATR=2.11 |
| Stop hit — per-position SL triggered | 2026-04-10 11:40:00 | 711.34 | 711.68 | 0.00 | SL hit |

### Cycle 99 — SELL (started 2026-04-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 11:05:00 | 741.90 | 746.76 | 0.00 | ORB-short ORB[748.15,753.85] vol=1.7x ATR=2.37 |
| Stop hit — per-position SL triggered | 2026-04-16 11:15:00 | 744.27 | 746.67 | 0.00 | SL hit |

### Cycle 100 — BUY (started 2026-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:30:00 | 762.15 | 759.47 | 0.00 | ORB-long ORB[753.95,761.40] vol=2.6x ATR=2.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:35:00 | 765.62 | 761.16 | 0.00 | T1 1.5R @ 765.62 |
| Target hit | 2026-04-21 10:55:00 | 766.55 | 766.75 | 0.00 | Trail-exit close<VWAP |

### Cycle 101 — SELL (started 2026-04-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 10:40:00 | 762.40 | 768.08 | 0.00 | ORB-short ORB[766.75,774.30] vol=1.5x ATR=2.35 |
| Stop hit — per-position SL triggered | 2026-04-23 11:05:00 | 764.75 | 767.72 | 0.00 | SL hit |

### Cycle 102 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:15:00 | 727.30 | 724.21 | 0.00 | ORB-long ORB[721.35,726.85] vol=2.7x ATR=2.01 |
| Stop hit — per-position SL triggered | 2026-05-04 11:30:00 | 725.29 | 724.54 | 0.00 | SL hit |

### Cycle 103 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 726.85 | 724.06 | 0.00 | ORB-long ORB[717.00,725.80] vol=5.3x ATR=3.06 |
| Stop hit — per-position SL triggered | 2026-05-05 10:10:00 | 723.79 | 725.90 | 0.00 | SL hit |

### Cycle 104 — SELL (started 2026-05-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:55:00 | 716.80 | 723.86 | 0.00 | ORB-short ORB[721.05,728.40] vol=3.0x ATR=2.02 |
| Stop hit — per-position SL triggered | 2026-05-06 11:35:00 | 718.82 | 722.57 | 0.00 | SL hit |

### Cycle 105 — SELL (started 2026-05-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:35:00 | 720.85 | 724.82 | 0.00 | ORB-short ORB[724.55,730.20] vol=2.3x ATR=2.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 09:45:00 | 716.96 | 722.58 | 0.00 | T1 1.5R @ 716.96 |
| Stop hit — per-position SL triggered | 2026-05-08 09:50:00 | 720.85 | 722.35 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-12 10:25:00 | 1213.50 | 2025-05-12 12:00:00 | 1203.65 | PARTIAL | 0.50 | 0.81% |
| SELL | retest1 | 2025-05-12 10:25:00 | 1213.50 | 2025-05-12 12:40:00 | 1211.00 | TARGET_HIT | 0.50 | 0.21% |
| BUY | retest1 | 2025-05-13 10:15:00 | 1226.80 | 2025-05-13 10:20:00 | 1233.31 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-05-13 10:15:00 | 1226.80 | 2025-05-13 10:25:00 | 1226.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-15 10:15:00 | 1218.10 | 2025-05-15 10:35:00 | 1213.88 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-05-19 09:35:00 | 1244.50 | 2025-05-19 10:10:00 | 1248.62 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-05-21 09:45:00 | 1233.50 | 2025-05-21 10:05:00 | 1239.57 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-05-21 09:45:00 | 1233.50 | 2025-05-21 15:20:00 | 1257.70 | TARGET_HIT | 0.50 | 1.96% |
| BUY | retest1 | 2025-05-23 10:25:00 | 1275.90 | 2025-05-23 10:30:00 | 1271.73 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-05-26 09:55:00 | 1292.60 | 2025-05-26 10:10:00 | 1287.69 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-05-30 11:15:00 | 1264.90 | 2025-05-30 13:40:00 | 1261.86 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-06-02 11:00:00 | 1269.10 | 2025-06-02 11:10:00 | 1266.51 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-06-04 11:15:00 | 1226.10 | 2025-06-04 13:40:00 | 1222.48 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-06-04 11:15:00 | 1226.10 | 2025-06-04 15:20:00 | 1222.40 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2025-06-09 09:45:00 | 1219.60 | 2025-06-09 15:20:00 | 1218.00 | TARGET_HIT | 1.00 | 0.13% |
| SELL | retest1 | 2025-06-11 11:15:00 | 1214.90 | 2025-06-11 12:35:00 | 1216.50 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2025-06-12 10:25:00 | 1225.10 | 2025-06-12 10:35:00 | 1222.87 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-06-17 10:15:00 | 1223.60 | 2025-06-17 10:45:00 | 1221.24 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-06-18 10:10:00 | 1221.30 | 2025-06-18 10:20:00 | 1219.33 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-06-25 09:35:00 | 1211.80 | 2025-06-25 09:50:00 | 1209.04 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-06-26 10:55:00 | 1209.00 | 2025-06-26 11:15:00 | 1210.92 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-07-02 11:10:00 | 1224.90 | 2025-07-02 11:30:00 | 1226.82 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-07-03 09:45:00 | 1234.10 | 2025-07-03 09:50:00 | 1231.50 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-07-07 09:35:00 | 1233.30 | 2025-07-07 09:45:00 | 1236.41 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-07-09 10:35:00 | 1262.50 | 2025-07-09 11:45:00 | 1258.42 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-07-10 10:00:00 | 1281.20 | 2025-07-10 10:10:00 | 1276.60 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-07-11 09:30:00 | 1254.20 | 2025-07-11 10:50:00 | 1247.73 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-07-11 09:30:00 | 1254.20 | 2025-07-11 15:20:00 | 1237.50 | TARGET_HIT | 0.50 | 1.33% |
| BUY | retest1 | 2025-07-15 11:00:00 | 1267.30 | 2025-07-15 11:10:00 | 1263.37 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-07-16 11:00:00 | 1250.10 | 2025-07-16 12:15:00 | 1246.07 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-07-16 11:00:00 | 1250.10 | 2025-07-16 15:20:00 | 1243.90 | TARGET_HIT | 0.50 | 0.50% |
| SELL | retest1 | 2025-07-17 09:45:00 | 1236.90 | 2025-07-17 14:25:00 | 1233.42 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-07-17 09:45:00 | 1236.90 | 2025-07-17 15:20:00 | 1229.40 | TARGET_HIT | 0.50 | 0.61% |
| SELL | retest1 | 2025-07-18 10:20:00 | 1219.70 | 2025-07-18 10:55:00 | 1222.24 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-07-23 10:55:00 | 1206.50 | 2025-07-23 11:30:00 | 1203.59 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-07-23 10:55:00 | 1206.50 | 2025-07-23 11:45:00 | 1206.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-24 10:00:00 | 1213.60 | 2025-07-24 10:05:00 | 1211.03 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-07-29 09:55:00 | 1189.00 | 2025-07-29 10:00:00 | 1192.16 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-08-05 11:15:00 | 1182.50 | 2025-08-05 11:50:00 | 1184.14 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-08-11 10:40:00 | 1189.30 | 2025-08-11 10:50:00 | 1186.65 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-08-13 09:30:00 | 1119.00 | 2025-08-13 10:30:00 | 1112.35 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2025-08-13 09:30:00 | 1119.00 | 2025-08-13 15:20:00 | 1091.90 | TARGET_HIT | 0.50 | 2.42% |
| SELL | retest1 | 2025-08-14 09:55:00 | 1083.90 | 2025-08-14 10:15:00 | 1078.70 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-08-14 09:55:00 | 1083.90 | 2025-08-14 11:05:00 | 1083.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-22 09:30:00 | 1106.20 | 2025-08-22 10:35:00 | 1109.74 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-08-25 10:15:00 | 1096.20 | 2025-08-25 11:45:00 | 1099.03 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-08-26 09:45:00 | 1084.80 | 2025-08-26 11:10:00 | 1087.66 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-08-28 09:35:00 | 1076.90 | 2025-08-28 09:45:00 | 1082.89 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-08-28 09:35:00 | 1076.90 | 2025-08-28 10:10:00 | 1076.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-01 10:50:00 | 1111.00 | 2025-09-01 11:00:00 | 1117.27 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-09-01 10:50:00 | 1111.00 | 2025-09-01 15:20:00 | 1115.90 | TARGET_HIT | 0.50 | 0.44% |
| BUY | retest1 | 2025-09-02 10:00:00 | 1134.40 | 2025-09-02 10:10:00 | 1140.31 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-09-02 10:00:00 | 1134.40 | 2025-09-02 14:15:00 | 1141.50 | TARGET_HIT | 0.50 | 0.63% |
| SELL | retest1 | 2025-09-11 10:20:00 | 1231.20 | 2025-09-11 10:25:00 | 1234.64 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-09-12 10:00:00 | 1256.10 | 2025-09-12 10:05:00 | 1251.69 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-09-16 10:05:00 | 1277.00 | 2025-09-16 10:15:00 | 1273.15 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-09-17 10:55:00 | 1264.20 | 2025-09-17 13:25:00 | 1260.19 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-09-17 10:55:00 | 1264.20 | 2025-09-17 15:20:00 | 1259.80 | TARGET_HIT | 0.50 | 0.35% |
| SELL | retest1 | 2025-09-19 09:55:00 | 1237.60 | 2025-09-19 10:00:00 | 1241.31 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-09-22 10:30:00 | 1224.30 | 2025-09-22 11:15:00 | 1227.81 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-09-25 09:40:00 | 1205.20 | 2025-09-25 09:50:00 | 1208.66 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-10-01 09:35:00 | 1189.20 | 2025-10-01 09:45:00 | 1183.94 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-10-03 09:50:00 | 1210.50 | 2025-10-03 10:00:00 | 1216.98 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-10-03 09:50:00 | 1210.50 | 2025-10-03 10:15:00 | 1210.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-06 10:30:00 | 1215.20 | 2025-10-06 10:45:00 | 1210.00 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-10-07 09:50:00 | 1231.60 | 2025-10-07 10:05:00 | 1226.49 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-10-08 09:55:00 | 1211.80 | 2025-10-08 10:10:00 | 1207.22 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-10-08 09:55:00 | 1211.80 | 2025-10-08 10:30:00 | 1211.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-10 09:35:00 | 1181.80 | 2025-10-10 09:55:00 | 1178.98 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-10-13 11:10:00 | 1158.90 | 2025-10-13 11:45:00 | 1155.48 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-10-13 11:10:00 | 1158.90 | 2025-10-13 15:20:00 | 1149.00 | TARGET_HIT | 0.50 | 0.85% |
| SELL | retest1 | 2025-10-14 09:35:00 | 1147.70 | 2025-10-14 09:50:00 | 1143.89 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-10-14 09:35:00 | 1147.70 | 2025-10-14 15:20:00 | 1125.20 | TARGET_HIT | 0.50 | 1.96% |
| SELL | retest1 | 2025-10-15 09:50:00 | 1112.00 | 2025-10-15 10:15:00 | 1114.83 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-10-20 09:40:00 | 1156.00 | 2025-10-20 09:45:00 | 1151.68 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-10-23 10:30:00 | 1175.20 | 2025-10-23 11:10:00 | 1172.29 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-10-30 10:25:00 | 1080.60 | 2025-10-30 11:25:00 | 1077.56 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-11-06 09:35:00 | 1052.60 | 2025-11-06 10:15:00 | 1048.95 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-11-06 09:35:00 | 1052.60 | 2025-11-06 12:00:00 | 1052.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-10 10:55:00 | 1048.50 | 2025-11-10 15:20:00 | 1049.60 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest1 | 2025-11-12 10:00:00 | 1045.70 | 2025-11-12 10:05:00 | 1047.55 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-11-21 09:50:00 | 1006.00 | 2025-11-21 09:55:00 | 1003.23 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-11-21 09:50:00 | 1006.00 | 2025-11-21 15:05:00 | 1001.00 | TARGET_HIT | 0.50 | 0.50% |
| BUY | retest1 | 2025-11-24 10:55:00 | 1009.80 | 2025-11-24 11:05:00 | 1007.55 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-12-01 09:50:00 | 991.25 | 2025-12-01 10:00:00 | 992.89 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-12-03 10:25:00 | 968.55 | 2025-12-03 11:05:00 | 966.05 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-12-05 10:50:00 | 963.20 | 2025-12-05 11:25:00 | 966.14 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-12-05 10:50:00 | 963.20 | 2025-12-05 12:00:00 | 963.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-08 11:10:00 | 954.00 | 2025-12-08 11:30:00 | 951.61 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-12-08 11:10:00 | 954.00 | 2025-12-08 11:40:00 | 954.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-09 10:05:00 | 953.10 | 2025-12-09 10:20:00 | 957.86 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-12-09 10:05:00 | 953.10 | 2025-12-09 15:20:00 | 983.50 | TARGET_HIT | 0.50 | 3.19% |
| SELL | retest1 | 2025-12-22 11:05:00 | 947.40 | 2025-12-22 11:10:00 | 945.09 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-12-22 11:05:00 | 947.40 | 2025-12-22 11:20:00 | 947.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-29 09:55:00 | 939.05 | 2025-12-29 10:15:00 | 936.22 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-12-29 09:55:00 | 939.05 | 2025-12-29 10:40:00 | 939.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-31 09:45:00 | 942.35 | 2025-12-31 09:50:00 | 940.77 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2026-01-06 09:30:00 | 953.00 | 2026-01-06 09:40:00 | 950.28 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2026-01-06 09:30:00 | 953.00 | 2026-01-06 09:45:00 | 953.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-08 09:40:00 | 933.00 | 2026-01-08 09:45:00 | 930.77 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2026-01-08 09:40:00 | 933.00 | 2026-01-08 14:55:00 | 928.05 | TARGET_HIT | 0.50 | 0.53% |
| SELL | retest1 | 2026-01-09 09:40:00 | 916.45 | 2026-01-09 09:45:00 | 918.68 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-01-12 10:35:00 | 909.30 | 2026-01-12 15:20:00 | 909.85 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest1 | 2026-01-14 10:45:00 | 907.05 | 2026-01-14 15:20:00 | 905.35 | TARGET_HIT | 1.00 | 0.19% |
| SELL | retest1 | 2026-01-16 09:35:00 | 900.00 | 2026-01-16 09:45:00 | 901.48 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2026-01-19 09:40:00 | 891.00 | 2026-01-19 10:10:00 | 893.28 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-01-20 10:45:00 | 902.65 | 2026-01-20 10:50:00 | 898.69 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-01-21 10:40:00 | 874.00 | 2026-01-21 11:25:00 | 868.90 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-01-21 10:40:00 | 874.00 | 2026-01-21 11:55:00 | 874.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-23 09:50:00 | 871.20 | 2026-01-23 10:20:00 | 874.02 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-01-30 09:30:00 | 861.65 | 2026-01-30 09:45:00 | 858.88 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-02-05 10:55:00 | 852.75 | 2026-02-05 12:30:00 | 854.24 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2026-02-06 11:00:00 | 844.20 | 2026-02-06 11:20:00 | 846.26 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-09 09:30:00 | 862.00 | 2026-02-09 09:35:00 | 860.13 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-13 11:15:00 | 866.05 | 2026-02-13 11:20:00 | 862.00 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-02-13 11:15:00 | 866.05 | 2026-02-13 12:05:00 | 866.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-16 11:05:00 | 841.00 | 2026-02-16 11:15:00 | 836.84 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-02-16 11:05:00 | 841.00 | 2026-02-16 15:20:00 | 828.95 | TARGET_HIT | 0.50 | 1.43% |
| BUY | retest1 | 2026-02-17 09:35:00 | 837.25 | 2026-02-17 09:40:00 | 834.44 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-02-19 11:15:00 | 832.60 | 2026-02-19 11:25:00 | 834.32 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-24 09:30:00 | 801.30 | 2026-02-24 10:05:00 | 804.24 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-27 10:15:00 | 793.25 | 2026-02-27 10:20:00 | 790.37 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-02-27 10:15:00 | 793.25 | 2026-02-27 12:25:00 | 793.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-04 09:30:00 | 753.85 | 2026-03-04 10:00:00 | 757.19 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-03-06 09:30:00 | 748.50 | 2026-03-06 10:00:00 | 745.32 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-03-13 09:50:00 | 705.85 | 2026-03-13 10:10:00 | 702.50 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-03-13 09:50:00 | 705.85 | 2026-03-13 11:50:00 | 703.30 | TARGET_HIT | 0.50 | 0.36% |
| BUY | retest1 | 2026-03-18 11:00:00 | 683.10 | 2026-03-18 11:05:00 | 680.18 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-04-09 09:30:00 | 697.50 | 2026-04-09 09:40:00 | 693.54 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-04-09 09:30:00 | 697.50 | 2026-04-09 10:40:00 | 697.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-10 10:50:00 | 713.45 | 2026-04-10 11:40:00 | 711.34 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-04-16 11:05:00 | 741.90 | 2026-04-16 11:15:00 | 744.27 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-21 09:30:00 | 762.15 | 2026-04-21 09:35:00 | 765.62 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-04-21 09:30:00 | 762.15 | 2026-04-21 10:55:00 | 766.55 | TARGET_HIT | 0.50 | 0.58% |
| SELL | retest1 | 2026-04-23 10:40:00 | 762.40 | 2026-04-23 11:05:00 | 764.75 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-05-04 11:15:00 | 727.30 | 2026-05-04 11:30:00 | 725.29 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-05-05 09:30:00 | 726.85 | 2026-05-05 10:10:00 | 723.79 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-05-06 10:55:00 | 716.80 | 2026-05-06 11:35:00 | 718.82 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-05-08 09:35:00 | 720.85 | 2026-05-08 09:45:00 | 716.96 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-05-08 09:35:00 | 720.85 | 2026-05-08 09:50:00 | 720.85 | STOP_HIT | 0.50 | 0.00% |
