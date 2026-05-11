# Narayana Hrudayalaya Ltd. (NH)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1820.00
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
| ENTRY1 | 89 |
| ENTRY2 | 0 |
| PARTIAL | 31 |
| TARGET_HIT | 20 |
| STOP_HIT | 69 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 120 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 51 / 69
- **Target hits / Stop hits / Partials:** 20 / 69 / 31
- **Avg / median % per leg:** 0.07% / 0.00%
- **Sum % (uncompounded):** 8.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 66 | 28 | 42.4% | 11 | 38 | 17 | 0.13% | 8.6% |
| BUY @ 2nd Alert (retest1) | 66 | 28 | 42.4% | 11 | 38 | 17 | 0.13% | 8.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 54 | 23 | 42.6% | 9 | 31 | 14 | -0.00% | -0.1% |
| SELL @ 2nd Alert (retest1) | 54 | 23 | 42.6% | 9 | 31 | 14 | -0.00% | -0.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 120 | 51 | 42.5% | 20 | 69 | 31 | 0.07% | 8.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 10:50:00 | 1257.00 | 1261.27 | 0.00 | ORB-short ORB[1263.05,1272.75] vol=2.0x ATR=5.49 |
| Stop hit — per-position SL triggered | 2024-05-13 11:30:00 | 1262.49 | 1261.64 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 10:40:00 | 1262.85 | 1257.30 | 0.00 | ORB-long ORB[1252.95,1262.70] vol=5.0x ATR=4.19 |
| Stop hit — per-position SL triggered | 2024-05-16 10:45:00 | 1258.66 | 1257.70 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 09:55:00 | 1280.00 | 1275.24 | 0.00 | ORB-long ORB[1267.20,1276.80] vol=1.6x ATR=4.12 |
| Stop hit — per-position SL triggered | 2024-05-17 10:05:00 | 1275.88 | 1275.52 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:40:00 | 1275.00 | 1279.90 | 0.00 | ORB-short ORB[1278.05,1287.85] vol=1.6x ATR=5.04 |
| Stop hit — per-position SL triggered | 2024-05-22 10:20:00 | 1280.04 | 1277.52 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 10:25:00 | 1250.10 | 1258.45 | 0.00 | ORB-short ORB[1251.05,1268.80] vol=4.3x ATR=4.99 |
| Stop hit — per-position SL triggered | 2024-05-24 11:00:00 | 1255.09 | 1255.80 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 09:30:00 | 1233.60 | 1239.94 | 0.00 | ORB-short ORB[1235.10,1253.60] vol=2.0x ATR=4.13 |
| Stop hit — per-position SL triggered | 2024-06-13 09:40:00 | 1237.73 | 1239.16 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:45:00 | 1248.20 | 1242.10 | 0.00 | ORB-long ORB[1227.20,1244.80] vol=2.1x ATR=3.87 |
| Stop hit — per-position SL triggered | 2024-06-14 10:10:00 | 1244.33 | 1243.00 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 09:30:00 | 1227.55 | 1233.04 | 0.00 | ORB-short ORB[1229.45,1242.75] vol=1.6x ATR=4.17 |
| Stop hit — per-position SL triggered | 2024-06-19 09:45:00 | 1231.72 | 1231.06 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 09:35:00 | 1220.00 | 1225.49 | 0.00 | ORB-short ORB[1223.80,1236.00] vol=1.7x ATR=4.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-21 10:25:00 | 1213.48 | 1221.69 | 0.00 | T1 1.5R @ 1213.48 |
| Target hit | 2024-06-21 12:10:00 | 1217.15 | 1216.84 | 0.00 | Trail-exit close>VWAP |

### Cycle 10 — BUY (started 2024-07-01 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 10:20:00 | 1219.15 | 1210.44 | 0.00 | ORB-long ORB[1202.40,1214.40] vol=1.6x ATR=4.25 |
| Stop hit — per-position SL triggered | 2024-07-01 10:30:00 | 1214.90 | 1211.37 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 09:35:00 | 1237.70 | 1232.97 | 0.00 | ORB-long ORB[1228.00,1236.95] vol=2.7x ATR=3.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 10:00:00 | 1243.30 | 1235.80 | 0.00 | T1 1.5R @ 1243.30 |
| Target hit | 2024-07-04 10:55:00 | 1241.25 | 1241.92 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — BUY (started 2024-07-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 10:40:00 | 1249.20 | 1239.96 | 0.00 | ORB-long ORB[1233.40,1245.65] vol=2.6x ATR=3.56 |
| Stop hit — per-position SL triggered | 2024-07-05 11:05:00 | 1245.64 | 1241.77 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 11:10:00 | 1238.10 | 1248.54 | 0.00 | ORB-short ORB[1244.25,1257.95] vol=1.7x ATR=3.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-08 12:00:00 | 1232.36 | 1246.97 | 0.00 | T1 1.5R @ 1232.36 |
| Stop hit — per-position SL triggered | 2024-07-08 15:20:00 | 1243.30 | 1235.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2024-07-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-10 10:50:00 | 1234.80 | 1226.60 | 0.00 | ORB-long ORB[1223.55,1232.80] vol=1.9x ATR=3.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 11:10:00 | 1240.54 | 1229.45 | 0.00 | T1 1.5R @ 1240.54 |
| Stop hit — per-position SL triggered | 2024-07-10 13:00:00 | 1234.80 | 1236.70 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 10:40:00 | 1242.50 | 1233.55 | 0.00 | ORB-long ORB[1228.00,1240.60] vol=2.2x ATR=4.44 |
| Stop hit — per-position SL triggered | 2024-07-11 11:05:00 | 1238.06 | 1236.85 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 09:30:00 | 1232.85 | 1237.64 | 0.00 | ORB-short ORB[1236.30,1244.00] vol=1.8x ATR=3.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 09:40:00 | 1228.02 | 1233.40 | 0.00 | T1 1.5R @ 1228.02 |
| Target hit | 2024-07-12 10:15:00 | 1230.95 | 1230.80 | 0.00 | Trail-exit close>VWAP |

### Cycle 17 — BUY (started 2024-07-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 10:00:00 | 1230.90 | 1215.49 | 0.00 | ORB-long ORB[1204.50,1217.95] vol=2.7x ATR=4.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 10:20:00 | 1237.19 | 1215.95 | 0.00 | T1 1.5R @ 1237.19 |
| Target hit | 2024-07-15 15:20:00 | 1247.55 | 1222.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — BUY (started 2024-07-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-19 09:40:00 | 1253.50 | 1248.48 | 0.00 | ORB-long ORB[1237.00,1247.95] vol=3.0x ATR=4.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 10:30:00 | 1259.79 | 1253.60 | 0.00 | T1 1.5R @ 1259.79 |
| Stop hit — per-position SL triggered | 2024-07-19 10:55:00 | 1253.50 | 1256.63 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 10:55:00 | 1246.80 | 1237.30 | 0.00 | ORB-long ORB[1223.00,1240.00] vol=2.1x ATR=4.23 |
| Stop hit — per-position SL triggered | 2024-07-22 11:00:00 | 1242.57 | 1237.49 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 09:45:00 | 1261.95 | 1256.99 | 0.00 | ORB-long ORB[1240.00,1258.20] vol=2.6x ATR=5.83 |
| Stop hit — per-position SL triggered | 2024-07-23 10:00:00 | 1256.12 | 1257.07 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-07-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-25 09:50:00 | 1237.25 | 1244.45 | 0.00 | ORB-short ORB[1238.95,1254.00] vol=3.6x ATR=4.20 |
| Stop hit — per-position SL triggered | 2024-07-25 09:55:00 | 1241.45 | 1243.52 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-07-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:45:00 | 1250.50 | 1242.09 | 0.00 | ORB-long ORB[1235.50,1246.25] vol=2.4x ATR=4.30 |
| Stop hit — per-position SL triggered | 2024-07-26 09:50:00 | 1246.20 | 1242.49 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-07-31 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-31 09:50:00 | 1263.95 | 1270.24 | 0.00 | ORB-short ORB[1267.60,1283.75] vol=1.7x ATR=4.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 13:20:00 | 1257.35 | 1265.61 | 0.00 | T1 1.5R @ 1257.35 |
| Target hit | 2024-07-31 15:20:00 | 1258.70 | 1260.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — SELL (started 2024-08-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-02 11:05:00 | 1236.35 | 1240.54 | 0.00 | ORB-short ORB[1238.30,1252.75] vol=3.2x ATR=4.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-02 12:00:00 | 1229.44 | 1237.96 | 0.00 | T1 1.5R @ 1229.44 |
| Target hit | 2024-08-02 15:20:00 | 1232.50 | 1234.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — SELL (started 2024-08-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 11:10:00 | 1236.90 | 1245.58 | 0.00 | ORB-short ORB[1239.15,1254.00] vol=2.5x ATR=4.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-06 11:35:00 | 1230.01 | 1244.25 | 0.00 | T1 1.5R @ 1230.01 |
| Stop hit — per-position SL triggered | 2024-08-06 12:10:00 | 1236.90 | 1242.60 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 09:55:00 | 1234.20 | 1227.62 | 0.00 | ORB-long ORB[1218.00,1234.00] vol=1.5x ATR=4.38 |
| Stop hit — per-position SL triggered | 2024-08-08 12:05:00 | 1229.82 | 1230.64 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-08-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 11:05:00 | 1201.05 | 1204.65 | 0.00 | ORB-short ORB[1201.50,1217.95] vol=3.4x ATR=2.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 11:25:00 | 1197.18 | 1203.90 | 0.00 | T1 1.5R @ 1197.18 |
| Target hit | 2024-08-13 15:15:00 | 1198.00 | 1197.89 | 0.00 | Trail-exit close>VWAP |

### Cycle 28 — SELL (started 2024-08-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-19 09:30:00 | 1179.00 | 1184.35 | 0.00 | ORB-short ORB[1180.10,1197.80] vol=1.8x ATR=3.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-19 10:00:00 | 1173.21 | 1179.44 | 0.00 | T1 1.5R @ 1173.21 |
| Stop hit — per-position SL triggered | 2024-08-19 11:15:00 | 1179.00 | 1177.06 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-08-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 09:55:00 | 1180.40 | 1184.71 | 0.00 | ORB-short ORB[1183.05,1194.80] vol=1.5x ATR=3.22 |
| Stop hit — per-position SL triggered | 2024-08-20 10:35:00 | 1183.62 | 1183.63 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 10:15:00 | 1274.00 | 1260.59 | 0.00 | ORB-long ORB[1247.00,1266.00] vol=2.9x ATR=7.51 |
| Stop hit — per-position SL triggered | 2024-08-22 15:15:00 | 1266.49 | 1265.88 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 11:15:00 | 1270.65 | 1279.66 | 0.00 | ORB-short ORB[1278.30,1289.75] vol=4.3x ATR=4.78 |
| Stop hit — per-position SL triggered | 2024-08-29 11:25:00 | 1275.43 | 1279.16 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-08-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 10:20:00 | 1289.50 | 1281.52 | 0.00 | ORB-long ORB[1268.00,1285.95] vol=5.2x ATR=5.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-30 10:30:00 | 1297.76 | 1285.52 | 0.00 | T1 1.5R @ 1297.76 |
| Stop hit — per-position SL triggered | 2024-08-30 10:50:00 | 1289.50 | 1294.10 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-09-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 10:40:00 | 1271.15 | 1274.02 | 0.00 | ORB-short ORB[1272.25,1280.00] vol=1.8x ATR=2.88 |
| Stop hit — per-position SL triggered | 2024-09-03 10:50:00 | 1274.03 | 1273.87 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-09-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 09:40:00 | 1302.75 | 1298.14 | 0.00 | ORB-long ORB[1285.30,1300.00] vol=3.2x ATR=4.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 09:55:00 | 1310.20 | 1302.92 | 0.00 | T1 1.5R @ 1310.20 |
| Target hit | 2024-09-05 10:15:00 | 1305.05 | 1307.84 | 0.00 | Trail-exit close<VWAP |

### Cycle 35 — SELL (started 2024-09-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-11 09:35:00 | 1342.00 | 1351.58 | 0.00 | ORB-short ORB[1346.80,1360.00] vol=1.6x ATR=5.18 |
| Stop hit — per-position SL triggered | 2024-09-11 10:15:00 | 1347.18 | 1348.59 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-09-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 11:05:00 | 1352.95 | 1345.15 | 0.00 | ORB-long ORB[1333.00,1348.40] vol=1.7x ATR=4.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 11:40:00 | 1358.99 | 1346.67 | 0.00 | T1 1.5R @ 1358.99 |
| Target hit | 2024-09-12 15:20:00 | 1364.25 | 1360.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2024-09-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 09:35:00 | 1280.95 | 1285.38 | 0.00 | ORB-short ORB[1283.05,1299.00] vol=4.7x ATR=4.28 |
| Stop hit — per-position SL triggered | 2024-09-17 09:40:00 | 1285.23 | 1285.39 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-09-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 10:05:00 | 1272.05 | 1281.81 | 0.00 | ORB-short ORB[1282.75,1294.35] vol=2.0x ATR=4.18 |
| Stop hit — per-position SL triggered | 2024-09-18 12:40:00 | 1276.23 | 1275.14 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 10:15:00 | 1248.60 | 1259.89 | 0.00 | ORB-short ORB[1259.00,1275.00] vol=2.3x ATR=4.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:35:00 | 1242.48 | 1255.31 | 0.00 | T1 1.5R @ 1242.48 |
| Stop hit — per-position SL triggered | 2024-09-19 12:15:00 | 1248.60 | 1247.40 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-09-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 09:35:00 | 1241.50 | 1235.07 | 0.00 | ORB-long ORB[1226.00,1237.95] vol=1.9x ATR=3.99 |
| Stop hit — per-position SL triggered | 2024-09-25 09:40:00 | 1237.51 | 1235.45 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-09-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-30 09:55:00 | 1225.35 | 1217.65 | 0.00 | ORB-long ORB[1208.00,1225.00] vol=1.9x ATR=3.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 10:20:00 | 1231.17 | 1220.13 | 0.00 | T1 1.5R @ 1231.17 |
| Target hit | 2024-09-30 13:20:00 | 1236.50 | 1237.61 | 0.00 | Trail-exit close<VWAP |

### Cycle 42 — BUY (started 2024-10-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 10:35:00 | 1235.05 | 1229.83 | 0.00 | ORB-long ORB[1220.10,1234.55] vol=1.7x ATR=3.56 |
| Stop hit — per-position SL triggered | 2024-10-09 11:50:00 | 1231.49 | 1232.21 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-10-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-15 11:05:00 | 1257.25 | 1265.29 | 0.00 | ORB-short ORB[1261.00,1277.95] vol=1.9x ATR=4.23 |
| Stop hit — per-position SL triggered | 2024-10-15 12:00:00 | 1261.48 | 1264.43 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-10-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 10:30:00 | 1279.60 | 1271.81 | 0.00 | ORB-long ORB[1263.80,1272.95] vol=6.4x ATR=4.64 |
| Stop hit — per-position SL triggered | 2024-10-16 10:45:00 | 1274.96 | 1273.43 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-10-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 11:10:00 | 1277.80 | 1282.67 | 0.00 | ORB-short ORB[1285.05,1295.55] vol=1.5x ATR=4.49 |
| Stop hit — per-position SL triggered | 2024-10-17 12:35:00 | 1282.29 | 1281.54 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-10-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-21 09:35:00 | 1281.80 | 1276.87 | 0.00 | ORB-long ORB[1262.20,1280.90] vol=2.7x ATR=4.55 |
| Stop hit — per-position SL triggered | 2024-10-21 13:05:00 | 1277.25 | 1279.56 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-10-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-24 11:10:00 | 1236.40 | 1243.03 | 0.00 | ORB-short ORB[1237.50,1248.60] vol=1.5x ATR=4.89 |
| Stop hit — per-position SL triggered | 2024-10-24 11:20:00 | 1241.29 | 1242.58 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-10-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 09:40:00 | 1230.00 | 1242.77 | 0.00 | ORB-short ORB[1242.00,1251.45] vol=2.9x ATR=5.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 10:05:00 | 1221.77 | 1233.13 | 0.00 | T1 1.5R @ 1221.77 |
| Stop hit — per-position SL triggered | 2024-10-29 12:05:00 | 1230.00 | 1228.73 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-11-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-05 09:45:00 | 1196.40 | 1202.45 | 0.00 | ORB-short ORB[1200.20,1211.05] vol=1.7x ATR=3.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-05 10:45:00 | 1190.66 | 1197.88 | 0.00 | T1 1.5R @ 1190.66 |
| Target hit | 2024-11-05 13:50:00 | 1195.55 | 1194.11 | 0.00 | Trail-exit close>VWAP |

### Cycle 50 — BUY (started 2024-11-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 09:50:00 | 1209.80 | 1197.74 | 0.00 | ORB-long ORB[1188.10,1200.95] vol=2.0x ATR=4.08 |
| Stop hit — per-position SL triggered | 2024-11-06 09:55:00 | 1205.72 | 1198.15 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-11-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-07 10:05:00 | 1259.55 | 1255.24 | 0.00 | ORB-long ORB[1249.60,1258.00] vol=1.6x ATR=5.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-07 10:55:00 | 1267.59 | 1260.72 | 0.00 | T1 1.5R @ 1267.59 |
| Target hit | 2024-11-07 15:20:00 | 1271.10 | 1266.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — BUY (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-08 10:15:00 | 1281.00 | 1271.26 | 0.00 | ORB-long ORB[1258.00,1274.85] vol=3.2x ATR=5.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-08 15:00:00 | 1288.61 | 1281.79 | 0.00 | T1 1.5R @ 1288.61 |
| Target hit | 2024-11-08 15:20:00 | 1289.45 | 1283.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — BUY (started 2024-11-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-12 09:50:00 | 1318.30 | 1309.80 | 0.00 | ORB-long ORB[1295.00,1314.00] vol=2.6x ATR=6.60 |
| Stop hit — per-position SL triggered | 2024-11-12 10:00:00 | 1311.70 | 1310.48 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-11-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-19 10:05:00 | 1257.95 | 1264.22 | 0.00 | ORB-short ORB[1258.35,1271.30] vol=2.1x ATR=3.53 |
| Stop hit — per-position SL triggered | 2024-11-19 10:10:00 | 1261.48 | 1263.84 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-11-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 09:45:00 | 1277.20 | 1272.06 | 0.00 | ORB-long ORB[1256.00,1275.05] vol=2.3x ATR=3.96 |
| Stop hit — per-position SL triggered | 2024-11-22 09:50:00 | 1273.24 | 1272.44 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-11-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:30:00 | 1257.25 | 1263.69 | 0.00 | ORB-short ORB[1258.65,1271.90] vol=3.0x ATR=3.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 10:50:00 | 1252.48 | 1262.21 | 0.00 | T1 1.5R @ 1252.48 |
| Target hit | 2024-11-28 15:00:00 | 1252.00 | 1251.96 | 0.00 | Trail-exit close>VWAP |

### Cycle 57 — SELL (started 2024-11-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-29 09:40:00 | 1250.15 | 1258.99 | 0.00 | ORB-short ORB[1253.55,1265.00] vol=1.8x ATR=5.18 |
| Stop hit — per-position SL triggered | 2024-11-29 09:45:00 | 1255.33 | 1258.71 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-12-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-02 11:05:00 | 1258.30 | 1263.29 | 0.00 | ORB-short ORB[1260.05,1274.20] vol=1.9x ATR=3.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-02 12:10:00 | 1253.00 | 1262.06 | 0.00 | T1 1.5R @ 1253.00 |
| Target hit | 2024-12-02 15:05:00 | 1258.00 | 1255.92 | 0.00 | Trail-exit close>VWAP |

### Cycle 59 — BUY (started 2024-12-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 10:10:00 | 1276.05 | 1266.17 | 0.00 | ORB-long ORB[1255.10,1270.00] vol=4.1x ATR=4.58 |
| Stop hit — per-position SL triggered | 2024-12-03 11:25:00 | 1271.47 | 1271.86 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2024-12-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 10:55:00 | 1292.10 | 1277.32 | 0.00 | ORB-long ORB[1269.80,1288.25] vol=2.3x ATR=5.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 11:15:00 | 1299.60 | 1278.37 | 0.00 | T1 1.5R @ 1299.60 |
| Target hit | 2024-12-04 15:20:00 | 1318.00 | 1291.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — BUY (started 2024-12-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 10:40:00 | 1321.45 | 1316.97 | 0.00 | ORB-long ORB[1305.10,1317.15] vol=1.9x ATR=4.73 |
| Stop hit — per-position SL triggered | 2024-12-06 12:15:00 | 1316.72 | 1317.70 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-12-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-10 09:40:00 | 1311.15 | 1316.86 | 0.00 | ORB-short ORB[1318.30,1335.00] vol=2.6x ATR=5.00 |
| Stop hit — per-position SL triggered | 2024-12-10 10:20:00 | 1316.15 | 1314.84 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 10:15:00 | 1295.65 | 1288.70 | 0.00 | ORB-long ORB[1275.10,1288.05] vol=2.3x ATR=4.37 |
| Stop hit — per-position SL triggered | 2024-12-12 12:45:00 | 1291.28 | 1292.48 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2024-12-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:00:00 | 1275.80 | 1278.86 | 0.00 | ORB-short ORB[1279.05,1294.90] vol=4.3x ATR=4.74 |
| Stop hit — per-position SL triggered | 2024-12-13 10:05:00 | 1280.54 | 1279.14 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2024-12-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-26 10:10:00 | 1314.90 | 1306.53 | 0.00 | ORB-long ORB[1299.00,1314.00] vol=2.4x ATR=5.04 |
| Stop hit — per-position SL triggered | 2024-12-26 10:15:00 | 1309.86 | 1307.46 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2024-12-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 10:05:00 | 1301.45 | 1305.56 | 0.00 | ORB-short ORB[1302.00,1316.85] vol=1.7x ATR=3.63 |
| Stop hit — per-position SL triggered | 2024-12-27 10:40:00 | 1305.08 | 1304.08 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-01-01 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 09:55:00 | 1292.50 | 1283.33 | 0.00 | ORB-long ORB[1267.10,1276.50] vol=5.7x ATR=5.91 |
| Stop hit — per-position SL triggered | 2025-01-01 10:00:00 | 1286.59 | 1283.87 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-01-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 09:50:00 | 1301.65 | 1306.06 | 0.00 | ORB-short ORB[1302.00,1318.00] vol=1.6x ATR=4.47 |
| Stop hit — per-position SL triggered | 2025-01-02 10:00:00 | 1306.12 | 1306.24 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-01-03 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 10:20:00 | 1310.95 | 1304.60 | 0.00 | ORB-long ORB[1299.65,1309.00] vol=1.9x ATR=4.23 |
| Stop hit — per-position SL triggered | 2025-01-03 12:40:00 | 1306.72 | 1307.31 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-01-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 10:20:00 | 1337.30 | 1324.80 | 0.00 | ORB-long ORB[1309.00,1327.00] vol=1.6x ATR=6.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-07 10:50:00 | 1347.02 | 1329.95 | 0.00 | T1 1.5R @ 1347.02 |
| Target hit | 2025-01-07 14:55:00 | 1347.85 | 1350.50 | 0.00 | Trail-exit close<VWAP |

### Cycle 71 — SELL (started 2025-01-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 09:40:00 | 1309.25 | 1317.33 | 0.00 | ORB-short ORB[1319.55,1338.80] vol=2.7x ATR=6.30 |
| Stop hit — per-position SL triggered | 2025-01-21 10:15:00 | 1315.55 | 1314.32 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2025-01-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 09:35:00 | 1309.10 | 1318.15 | 0.00 | ORB-short ORB[1319.70,1330.95] vol=3.3x ATR=4.87 |
| Stop hit — per-position SL triggered | 2025-01-24 10:25:00 | 1313.97 | 1312.43 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2025-01-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 11:00:00 | 1277.85 | 1272.91 | 0.00 | ORB-long ORB[1265.20,1276.35] vol=3.6x ATR=3.23 |
| Stop hit — per-position SL triggered | 2025-01-29 11:35:00 | 1274.62 | 1273.83 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2025-01-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 10:45:00 | 1321.55 | 1310.81 | 0.00 | ORB-long ORB[1296.15,1315.00] vol=6.4x ATR=5.01 |
| Stop hit — per-position SL triggered | 2025-01-30 11:00:00 | 1316.54 | 1312.61 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2025-01-31 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 10:20:00 | 1320.70 | 1314.33 | 0.00 | ORB-long ORB[1300.05,1313.05] vol=1.7x ATR=5.27 |
| Stop hit — per-position SL triggered | 2025-01-31 11:05:00 | 1315.43 | 1316.43 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2025-02-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-04 09:40:00 | 1371.95 | 1361.74 | 0.00 | ORB-long ORB[1346.40,1363.95] vol=1.5x ATR=5.43 |
| Stop hit — per-position SL triggered | 2025-02-04 09:50:00 | 1366.52 | 1363.19 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2025-02-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-06 09:40:00 | 1373.15 | 1367.38 | 0.00 | ORB-long ORB[1360.05,1371.80] vol=2.9x ATR=5.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 10:10:00 | 1380.76 | 1373.29 | 0.00 | T1 1.5R @ 1380.76 |
| Target hit | 2025-02-06 11:05:00 | 1388.50 | 1392.68 | 0.00 | Trail-exit close<VWAP |

### Cycle 78 — BUY (started 2025-02-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-19 09:50:00 | 1373.25 | 1364.31 | 0.00 | ORB-long ORB[1352.40,1370.00] vol=1.7x ATR=6.05 |
| Stop hit — per-position SL triggered | 2025-02-19 10:10:00 | 1367.20 | 1366.12 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2025-02-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 09:35:00 | 1391.95 | 1379.79 | 0.00 | ORB-long ORB[1370.00,1380.90] vol=4.5x ATR=4.61 |
| Stop hit — per-position SL triggered | 2025-02-20 09:40:00 | 1387.34 | 1381.14 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2025-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 09:30:00 | 1406.80 | 1397.42 | 0.00 | ORB-long ORB[1387.50,1399.00] vol=1.7x ATR=5.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 09:45:00 | 1415.59 | 1409.72 | 0.00 | T1 1.5R @ 1415.59 |
| Target hit | 2025-02-25 10:20:00 | 1431.30 | 1433.60 | 0.00 | Trail-exit close<VWAP |

### Cycle 81 — SELL (started 2025-03-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-11 09:35:00 | 1562.15 | 1568.95 | 0.00 | ORB-short ORB[1563.00,1586.45] vol=2.3x ATR=9.18 |
| Stop hit — per-position SL triggered | 2025-03-11 09:40:00 | 1571.33 | 1568.95 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2025-03-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-12 09:35:00 | 1598.90 | 1595.73 | 0.00 | ORB-long ORB[1576.05,1597.70] vol=2.2x ATR=8.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 09:50:00 | 1611.67 | 1598.64 | 0.00 | T1 1.5R @ 1611.67 |
| Stop hit — per-position SL triggered | 2025-03-12 10:20:00 | 1598.90 | 1600.74 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2025-03-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:40:00 | 1602.55 | 1590.61 | 0.00 | ORB-long ORB[1575.45,1592.90] vol=3.0x ATR=7.77 |
| Stop hit — per-position SL triggered | 2025-03-18 09:55:00 | 1594.78 | 1592.40 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2025-03-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-20 09:30:00 | 1649.05 | 1655.82 | 0.00 | ORB-short ORB[1651.05,1665.00] vol=1.7x ATR=6.31 |
| Stop hit — per-position SL triggered | 2025-03-20 09:50:00 | 1655.36 | 1652.21 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2025-04-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-03 09:45:00 | 1663.15 | 1679.69 | 0.00 | ORB-short ORB[1667.00,1690.45] vol=2.0x ATR=8.11 |
| Stop hit — per-position SL triggered | 2025-04-03 09:50:00 | 1671.26 | 1678.48 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2025-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 09:35:00 | 1823.70 | 1837.52 | 0.00 | ORB-short ORB[1837.00,1851.00] vol=2.5x ATR=6.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 10:05:00 | 1813.37 | 1830.84 | 0.00 | T1 1.5R @ 1813.37 |
| Target hit | 2025-04-23 15:20:00 | 1800.80 | 1810.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 87 — BUY (started 2025-05-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 11:05:00 | 1781.50 | 1778.72 | 0.00 | ORB-long ORB[1760.90,1780.70] vol=1.6x ATR=5.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 12:00:00 | 1789.50 | 1779.98 | 0.00 | T1 1.5R @ 1789.50 |
| Stop hit — per-position SL triggered | 2025-05-05 13:05:00 | 1781.50 | 1783.29 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2025-05-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-07 11:10:00 | 1810.00 | 1793.86 | 0.00 | ORB-long ORB[1765.00,1785.50] vol=1.7x ATR=6.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 11:35:00 | 1820.33 | 1795.23 | 0.00 | T1 1.5R @ 1820.33 |
| Stop hit — per-position SL triggered | 2025-05-07 12:25:00 | 1810.00 | 1798.93 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2025-05-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 09:30:00 | 1840.90 | 1830.84 | 0.00 | ORB-long ORB[1816.10,1836.00] vol=2.9x ATR=6.34 |
| Stop hit — per-position SL triggered | 2025-05-08 09:45:00 | 1834.56 | 1836.57 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-13 10:50:00 | 1257.00 | 2024-05-13 11:30:00 | 1262.49 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-05-16 10:40:00 | 1262.85 | 2024-05-16 10:45:00 | 1258.66 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-05-17 09:55:00 | 1280.00 | 2024-05-17 10:05:00 | 1275.88 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-05-22 09:40:00 | 1275.00 | 2024-05-22 10:20:00 | 1280.04 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-05-24 10:25:00 | 1250.10 | 2024-05-24 11:00:00 | 1255.09 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-06-13 09:30:00 | 1233.60 | 2024-06-13 09:40:00 | 1237.73 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-06-14 09:45:00 | 1248.20 | 2024-06-14 10:10:00 | 1244.33 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-06-19 09:30:00 | 1227.55 | 2024-06-19 09:45:00 | 1231.72 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-06-21 09:35:00 | 1220.00 | 2024-06-21 10:25:00 | 1213.48 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-06-21 09:35:00 | 1220.00 | 2024-06-21 12:10:00 | 1217.15 | TARGET_HIT | 0.50 | 0.23% |
| BUY | retest1 | 2024-07-01 10:20:00 | 1219.15 | 2024-07-01 10:30:00 | 1214.90 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-07-04 09:35:00 | 1237.70 | 2024-07-04 10:00:00 | 1243.30 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-07-04 09:35:00 | 1237.70 | 2024-07-04 10:55:00 | 1241.25 | TARGET_HIT | 0.50 | 0.29% |
| BUY | retest1 | 2024-07-05 10:40:00 | 1249.20 | 2024-07-05 11:05:00 | 1245.64 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-07-08 11:10:00 | 1238.10 | 2024-07-08 12:00:00 | 1232.36 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-07-08 11:10:00 | 1238.10 | 2024-07-08 15:20:00 | 1243.30 | STOP_HIT | 0.50 | -0.42% |
| BUY | retest1 | 2024-07-10 10:50:00 | 1234.80 | 2024-07-10 11:10:00 | 1240.54 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-07-10 10:50:00 | 1234.80 | 2024-07-10 13:00:00 | 1234.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-11 10:40:00 | 1242.50 | 2024-07-11 11:05:00 | 1238.06 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-07-12 09:30:00 | 1232.85 | 2024-07-12 09:40:00 | 1228.02 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-07-12 09:30:00 | 1232.85 | 2024-07-12 10:15:00 | 1230.95 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2024-07-15 10:00:00 | 1230.90 | 2024-07-15 10:20:00 | 1237.19 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-07-15 10:00:00 | 1230.90 | 2024-07-15 15:20:00 | 1247.55 | TARGET_HIT | 0.50 | 1.35% |
| BUY | retest1 | 2024-07-19 09:40:00 | 1253.50 | 2024-07-19 10:30:00 | 1259.79 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-07-19 09:40:00 | 1253.50 | 2024-07-19 10:55:00 | 1253.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-22 10:55:00 | 1246.80 | 2024-07-22 11:00:00 | 1242.57 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-07-23 09:45:00 | 1261.95 | 2024-07-23 10:00:00 | 1256.12 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-07-25 09:50:00 | 1237.25 | 2024-07-25 09:55:00 | 1241.45 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-07-26 09:45:00 | 1250.50 | 2024-07-26 09:50:00 | 1246.20 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-07-31 09:50:00 | 1263.95 | 2024-07-31 13:20:00 | 1257.35 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-07-31 09:50:00 | 1263.95 | 2024-07-31 15:20:00 | 1258.70 | TARGET_HIT | 0.50 | 0.42% |
| SELL | retest1 | 2024-08-02 11:05:00 | 1236.35 | 2024-08-02 12:00:00 | 1229.44 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-08-02 11:05:00 | 1236.35 | 2024-08-02 15:20:00 | 1232.50 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2024-08-06 11:10:00 | 1236.90 | 2024-08-06 11:35:00 | 1230.01 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-08-06 11:10:00 | 1236.90 | 2024-08-06 12:10:00 | 1236.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-08 09:55:00 | 1234.20 | 2024-08-08 12:05:00 | 1229.82 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-08-13 11:05:00 | 1201.05 | 2024-08-13 11:25:00 | 1197.18 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-08-13 11:05:00 | 1201.05 | 2024-08-13 15:15:00 | 1198.00 | TARGET_HIT | 0.50 | 0.25% |
| SELL | retest1 | 2024-08-19 09:30:00 | 1179.00 | 2024-08-19 10:00:00 | 1173.21 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-08-19 09:30:00 | 1179.00 | 2024-08-19 11:15:00 | 1179.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-20 09:55:00 | 1180.40 | 2024-08-20 10:35:00 | 1183.62 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-08-22 10:15:00 | 1274.00 | 2024-08-22 15:15:00 | 1266.49 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest1 | 2024-08-29 11:15:00 | 1270.65 | 2024-08-29 11:25:00 | 1275.43 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-08-30 10:20:00 | 1289.50 | 2024-08-30 10:30:00 | 1297.76 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-08-30 10:20:00 | 1289.50 | 2024-08-30 10:50:00 | 1289.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-03 10:40:00 | 1271.15 | 2024-09-03 10:50:00 | 1274.03 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-09-05 09:40:00 | 1302.75 | 2024-09-05 09:55:00 | 1310.20 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-09-05 09:40:00 | 1302.75 | 2024-09-05 10:15:00 | 1305.05 | TARGET_HIT | 0.50 | 0.18% |
| SELL | retest1 | 2024-09-11 09:35:00 | 1342.00 | 2024-09-11 10:15:00 | 1347.18 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-09-12 11:05:00 | 1352.95 | 2024-09-12 11:40:00 | 1358.99 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-09-12 11:05:00 | 1352.95 | 2024-09-12 15:20:00 | 1364.25 | TARGET_HIT | 0.50 | 0.84% |
| SELL | retest1 | 2024-09-17 09:35:00 | 1280.95 | 2024-09-17 09:40:00 | 1285.23 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-09-18 10:05:00 | 1272.05 | 2024-09-18 12:40:00 | 1276.23 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-09-19 10:15:00 | 1248.60 | 2024-09-19 10:35:00 | 1242.48 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-09-19 10:15:00 | 1248.60 | 2024-09-19 12:15:00 | 1248.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-25 09:35:00 | 1241.50 | 2024-09-25 09:40:00 | 1237.51 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-09-30 09:55:00 | 1225.35 | 2024-09-30 10:20:00 | 1231.17 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-09-30 09:55:00 | 1225.35 | 2024-09-30 13:20:00 | 1236.50 | TARGET_HIT | 0.50 | 0.91% |
| BUY | retest1 | 2024-10-09 10:35:00 | 1235.05 | 2024-10-09 11:50:00 | 1231.49 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-10-15 11:05:00 | 1257.25 | 2024-10-15 12:00:00 | 1261.48 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-10-16 10:30:00 | 1279.60 | 2024-10-16 10:45:00 | 1274.96 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-10-17 11:10:00 | 1277.80 | 2024-10-17 12:35:00 | 1282.29 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-10-21 09:35:00 | 1281.80 | 2024-10-21 13:05:00 | 1277.25 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-10-24 11:10:00 | 1236.40 | 2024-10-24 11:20:00 | 1241.29 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-10-29 09:40:00 | 1230.00 | 2024-10-29 10:05:00 | 1221.77 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2024-10-29 09:40:00 | 1230.00 | 2024-10-29 12:05:00 | 1230.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-05 09:45:00 | 1196.40 | 2024-11-05 10:45:00 | 1190.66 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-11-05 09:45:00 | 1196.40 | 2024-11-05 13:50:00 | 1195.55 | TARGET_HIT | 0.50 | 0.07% |
| BUY | retest1 | 2024-11-06 09:50:00 | 1209.80 | 2024-11-06 09:55:00 | 1205.72 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-11-07 10:05:00 | 1259.55 | 2024-11-07 10:55:00 | 1267.59 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-11-07 10:05:00 | 1259.55 | 2024-11-07 15:20:00 | 1271.10 | TARGET_HIT | 0.50 | 0.92% |
| BUY | retest1 | 2024-11-08 10:15:00 | 1281.00 | 2024-11-08 15:00:00 | 1288.61 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-11-08 10:15:00 | 1281.00 | 2024-11-08 15:20:00 | 1289.45 | TARGET_HIT | 0.50 | 0.66% |
| BUY | retest1 | 2024-11-12 09:50:00 | 1318.30 | 2024-11-12 10:00:00 | 1311.70 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-11-19 10:05:00 | 1257.95 | 2024-11-19 10:10:00 | 1261.48 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-11-22 09:45:00 | 1277.20 | 2024-11-22 09:50:00 | 1273.24 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-11-28 10:30:00 | 1257.25 | 2024-11-28 10:50:00 | 1252.48 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-11-28 10:30:00 | 1257.25 | 2024-11-28 15:00:00 | 1252.00 | TARGET_HIT | 0.50 | 0.42% |
| SELL | retest1 | 2024-11-29 09:40:00 | 1250.15 | 2024-11-29 09:45:00 | 1255.33 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-12-02 11:05:00 | 1258.30 | 2024-12-02 12:10:00 | 1253.00 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-12-02 11:05:00 | 1258.30 | 2024-12-02 15:05:00 | 1258.00 | TARGET_HIT | 0.50 | 0.02% |
| BUY | retest1 | 2024-12-03 10:10:00 | 1276.05 | 2024-12-03 11:25:00 | 1271.47 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-12-04 10:55:00 | 1292.10 | 2024-12-04 11:15:00 | 1299.60 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-12-04 10:55:00 | 1292.10 | 2024-12-04 15:20:00 | 1318.00 | TARGET_HIT | 0.50 | 2.00% |
| BUY | retest1 | 2024-12-06 10:40:00 | 1321.45 | 2024-12-06 12:15:00 | 1316.72 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-12-10 09:40:00 | 1311.15 | 2024-12-10 10:20:00 | 1316.15 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-12-12 10:15:00 | 1295.65 | 2024-12-12 12:45:00 | 1291.28 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-12-13 10:00:00 | 1275.80 | 2024-12-13 10:05:00 | 1280.54 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-12-26 10:10:00 | 1314.90 | 2024-12-26 10:15:00 | 1309.86 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-12-27 10:05:00 | 1301.45 | 2024-12-27 10:40:00 | 1305.08 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-01-01 09:55:00 | 1292.50 | 2025-01-01 10:00:00 | 1286.59 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-01-02 09:50:00 | 1301.65 | 2025-01-02 10:00:00 | 1306.12 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-01-03 10:20:00 | 1310.95 | 2025-01-03 12:40:00 | 1306.72 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-01-07 10:20:00 | 1337.30 | 2025-01-07 10:50:00 | 1347.02 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2025-01-07 10:20:00 | 1337.30 | 2025-01-07 14:55:00 | 1347.85 | TARGET_HIT | 0.50 | 0.79% |
| SELL | retest1 | 2025-01-21 09:40:00 | 1309.25 | 2025-01-21 10:15:00 | 1315.55 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2025-01-24 09:35:00 | 1309.10 | 2025-01-24 10:25:00 | 1313.97 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-01-29 11:00:00 | 1277.85 | 2025-01-29 11:35:00 | 1274.62 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-01-30 10:45:00 | 1321.55 | 2025-01-30 11:00:00 | 1316.54 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-01-31 10:20:00 | 1320.70 | 2025-01-31 11:05:00 | 1315.43 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-02-04 09:40:00 | 1371.95 | 2025-02-04 09:50:00 | 1366.52 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-02-06 09:40:00 | 1373.15 | 2025-02-06 10:10:00 | 1380.76 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-02-06 09:40:00 | 1373.15 | 2025-02-06 11:05:00 | 1388.50 | TARGET_HIT | 0.50 | 1.12% |
| BUY | retest1 | 2025-02-19 09:50:00 | 1373.25 | 2025-02-19 10:10:00 | 1367.20 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-02-20 09:35:00 | 1391.95 | 2025-02-20 09:40:00 | 1387.34 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-02-25 09:30:00 | 1406.80 | 2025-02-25 09:45:00 | 1415.59 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-02-25 09:30:00 | 1406.80 | 2025-02-25 10:20:00 | 1431.30 | TARGET_HIT | 0.50 | 1.74% |
| SELL | retest1 | 2025-03-11 09:35:00 | 1562.15 | 2025-03-11 09:40:00 | 1571.33 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest1 | 2025-03-12 09:35:00 | 1598.90 | 2025-03-12 09:50:00 | 1611.67 | PARTIAL | 0.50 | 0.80% |
| BUY | retest1 | 2025-03-12 09:35:00 | 1598.90 | 2025-03-12 10:20:00 | 1598.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-18 09:40:00 | 1602.55 | 2025-03-18 09:55:00 | 1594.78 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2025-03-20 09:30:00 | 1649.05 | 2025-03-20 09:50:00 | 1655.36 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-04-03 09:45:00 | 1663.15 | 2025-04-03 09:50:00 | 1671.26 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2025-04-23 09:35:00 | 1823.70 | 2025-04-23 10:05:00 | 1813.37 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-04-23 09:35:00 | 1823.70 | 2025-04-23 15:20:00 | 1800.80 | TARGET_HIT | 0.50 | 1.26% |
| BUY | retest1 | 2025-05-05 11:05:00 | 1781.50 | 2025-05-05 12:00:00 | 1789.50 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-05-05 11:05:00 | 1781.50 | 2025-05-05 13:05:00 | 1781.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-07 11:10:00 | 1810.00 | 2025-05-07 11:35:00 | 1820.33 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-05-07 11:10:00 | 1810.00 | 2025-05-07 12:25:00 | 1810.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-08 09:30:00 | 1840.90 | 2025-05-08 09:45:00 | 1834.56 | STOP_HIT | 1.00 | -0.34% |
