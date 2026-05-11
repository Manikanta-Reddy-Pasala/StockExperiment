# NESTLEIND (NESTLEIND)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (35296 bars)
- **Last close:** 1475.30
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
| ENTRY1 | 96 |
| ENTRY2 | 0 |
| PARTIAL | 46 |
| TARGET_HIT | 13 |
| STOP_HIT | 83 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 142 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 59 / 83
- **Target hits / Stop hits / Partials:** 13 / 83 / 46
- **Avg / median % per leg:** 0.09% / 0.00%
- **Sum % (uncompounded):** 12.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 76 | 31 | 40.8% | 6 | 45 | 25 | 0.09% | 6.6% |
| BUY @ 2nd Alert (retest1) | 76 | 31 | 40.8% | 6 | 45 | 25 | 0.09% | 6.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 66 | 28 | 42.4% | 7 | 38 | 21 | 0.09% | 5.9% |
| SELL @ 2nd Alert (retest1) | 66 | 28 | 42.4% | 7 | 38 | 21 | 0.09% | 5.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 142 | 59 | 41.5% | 13 | 83 | 46 | 0.09% | 12.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-14 09:30:00 | 1244.43 | 1250.19 | 0.00 | ORB-short ORB[1246.00,1259.47] vol=1.7x ATR=3.17 |
| Stop hit — per-position SL triggered | 2024-05-14 09:45:00 | 1247.60 | 1248.65 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 09:30:00 | 1226.30 | 1230.83 | 0.00 | ORB-short ORB[1228.88,1239.97] vol=2.0x ATR=2.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 10:20:00 | 1222.40 | 1225.48 | 0.00 | T1 1.5R @ 1222.40 |
| Target hit | 2024-05-16 12:05:00 | 1223.15 | 1222.23 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — BUY (started 2024-05-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-18 09:45:00 | 1257.97 | 1244.17 | 0.00 | ORB-long ORB[1229.90,1234.00] vol=3.8x ATR=3.91 |
| Stop hit — per-position SL triggered | 2024-05-18 09:50:00 | 1254.06 | 1247.20 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 10:45:00 | 1235.85 | 1230.68 | 0.00 | ORB-long ORB[1222.50,1231.08] vol=2.3x ATR=2.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-29 11:10:00 | 1239.87 | 1232.64 | 0.00 | T1 1.5R @ 1239.87 |
| Stop hit — per-position SL triggered | 2024-05-29 11:20:00 | 1235.85 | 1232.93 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 10:35:00 | 1245.35 | 1242.00 | 0.00 | ORB-long ORB[1233.50,1244.20] vol=5.9x ATR=3.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-07 11:35:00 | 1250.89 | 1243.63 | 0.00 | T1 1.5R @ 1250.89 |
| Stop hit — per-position SL triggered | 2024-06-07 13:00:00 | 1245.35 | 1246.72 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 11:15:00 | 1275.00 | 1265.49 | 0.00 | ORB-long ORB[1253.53,1264.90] vol=3.8x ATR=3.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 11:45:00 | 1279.84 | 1267.54 | 0.00 | T1 1.5R @ 1279.84 |
| Stop hit — per-position SL triggered | 2024-06-10 14:35:00 | 1275.00 | 1272.48 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 10:05:00 | 1263.22 | 1268.61 | 0.00 | ORB-short ORB[1267.53,1277.38] vol=2.1x ATR=3.29 |
| Stop hit — per-position SL triggered | 2024-06-12 10:15:00 | 1266.51 | 1268.12 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-14 11:15:00 | 1271.95 | 1273.30 | 0.00 | ORB-short ORB[1272.50,1282.30] vol=2.2x ATR=2.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 11:30:00 | 1268.60 | 1273.12 | 0.00 | T1 1.5R @ 1268.60 |
| Stop hit — per-position SL triggered | 2024-06-14 13:05:00 | 1271.95 | 1272.50 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-18 11:05:00 | 1271.00 | 1273.08 | 0.00 | ORB-short ORB[1271.25,1277.50] vol=1.6x ATR=2.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 12:30:00 | 1267.72 | 1272.23 | 0.00 | T1 1.5R @ 1267.72 |
| Stop hit — per-position SL triggered | 2024-06-18 13:10:00 | 1271.00 | 1272.03 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 10:40:00 | 1254.43 | 1262.32 | 0.00 | ORB-short ORB[1261.88,1274.00] vol=1.6x ATR=3.01 |
| Stop hit — per-position SL triggered | 2024-06-21 11:35:00 | 1257.44 | 1259.05 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:55:00 | 1265.38 | 1260.24 | 0.00 | ORB-long ORB[1254.38,1263.50] vol=3.1x ATR=2.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 12:00:00 | 1268.67 | 1262.41 | 0.00 | T1 1.5R @ 1268.67 |
| Stop hit — per-position SL triggered | 2024-06-26 15:00:00 | 1265.38 | 1266.27 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 11:15:00 | 1284.65 | 1280.43 | 0.00 | ORB-long ORB[1264.15,1282.10] vol=2.9x ATR=2.91 |
| Stop hit — per-position SL triggered | 2024-06-28 11:45:00 | 1281.74 | 1280.83 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 10:40:00 | 1293.03 | 1285.87 | 0.00 | ORB-long ORB[1278.50,1288.75] vol=2.7x ATR=3.16 |
| Stop hit — per-position SL triggered | 2024-07-01 11:25:00 | 1289.87 | 1286.98 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 11:15:00 | 1281.10 | 1280.03 | 0.00 | ORB-long ORB[1270.58,1281.00] vol=2.6x ATR=2.38 |
| Stop hit — per-position SL triggered | 2024-07-03 11:25:00 | 1278.72 | 1280.01 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 10:00:00 | 1300.03 | 1290.05 | 0.00 | ORB-long ORB[1283.13,1290.00] vol=1.8x ATR=3.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-08 10:10:00 | 1305.79 | 1293.94 | 0.00 | T1 1.5R @ 1305.79 |
| Stop hit — per-position SL triggered | 2024-07-08 10:30:00 | 1300.03 | 1296.31 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 09:35:00 | 1296.18 | 1301.96 | 0.00 | ORB-short ORB[1301.33,1315.80] vol=1.8x ATR=4.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 10:10:00 | 1289.47 | 1297.45 | 0.00 | T1 1.5R @ 1289.47 |
| Target hit | 2024-07-11 12:15:00 | 1293.33 | 1292.82 | 0.00 | Trail-exit close>VWAP |

### Cycle 17 — BUY (started 2024-07-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 10:55:00 | 1305.50 | 1300.42 | 0.00 | ORB-long ORB[1297.50,1302.72] vol=3.3x ATR=3.73 |
| Stop hit — per-position SL triggered | 2024-07-12 11:10:00 | 1301.77 | 1302.21 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-16 10:20:00 | 1289.25 | 1296.60 | 0.00 | ORB-short ORB[1292.60,1302.30] vol=2.9x ATR=3.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 10:40:00 | 1284.65 | 1293.94 | 0.00 | T1 1.5R @ 1284.65 |
| Stop hit — per-position SL triggered | 2024-07-16 10:45:00 | 1289.25 | 1293.78 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-22 10:55:00 | 1295.00 | 1305.90 | 0.00 | ORB-short ORB[1298.93,1315.00] vol=2.2x ATR=3.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 12:05:00 | 1289.71 | 1302.45 | 0.00 | T1 1.5R @ 1289.71 |
| Stop hit — per-position SL triggered | 2024-07-22 13:25:00 | 1295.00 | 1297.14 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-07-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-25 09:40:00 | 1268.72 | 1271.79 | 0.00 | ORB-short ORB[1270.55,1278.33] vol=2.5x ATR=3.80 |
| Stop hit — per-position SL triggered | 2024-07-25 10:15:00 | 1272.52 | 1271.19 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-07-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-30 10:20:00 | 1227.95 | 1231.76 | 0.00 | ORB-short ORB[1232.50,1240.20] vol=2.1x ATR=2.06 |
| Stop hit — per-position SL triggered | 2024-07-30 10:25:00 | 1230.01 | 1231.66 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 10:30:00 | 1242.50 | 1237.83 | 0.00 | ORB-long ORB[1228.18,1235.55] vol=1.5x ATR=2.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 10:35:00 | 1246.12 | 1240.71 | 0.00 | T1 1.5R @ 1246.12 |
| Stop hit — per-position SL triggered | 2024-08-01 10:40:00 | 1242.50 | 1240.76 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-02 11:00:00 | 1236.03 | 1242.70 | 0.00 | ORB-short ORB[1237.50,1251.13] vol=1.5x ATR=3.67 |
| Stop hit — per-position SL triggered | 2024-08-02 11:10:00 | 1239.70 | 1242.57 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-05 10:00:00 | 1262.45 | 1247.73 | 0.00 | ORB-long ORB[1237.53,1247.45] vol=1.5x ATR=4.15 |
| Stop hit — per-position SL triggered | 2024-08-05 10:10:00 | 1258.30 | 1249.38 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-08 10:55:00 | 1246.20 | 1254.75 | 0.00 | ORB-short ORB[1254.25,1262.50] vol=2.1x ATR=2.48 |
| Stop hit — per-position SL triggered | 2024-08-08 11:15:00 | 1248.68 | 1250.58 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 10:05:00 | 1272.55 | 1269.84 | 0.00 | ORB-long ORB[1260.83,1267.15] vol=1.8x ATR=2.68 |
| Stop hit — per-position SL triggered | 2024-08-21 10:35:00 | 1269.87 | 1270.17 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-08-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 10:35:00 | 1258.10 | 1260.46 | 0.00 | ORB-short ORB[1260.53,1266.40] vol=4.5x ATR=2.34 |
| Stop hit — per-position SL triggered | 2024-08-27 10:45:00 | 1260.44 | 1260.37 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-08-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 10:45:00 | 1250.00 | 1253.50 | 0.00 | ORB-short ORB[1253.83,1259.50] vol=2.8x ATR=2.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-28 11:05:00 | 1247.01 | 1252.68 | 0.00 | T1 1.5R @ 1247.01 |
| Stop hit — per-position SL triggered | 2024-08-28 11:15:00 | 1250.00 | 1252.42 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-08-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 10:55:00 | 1251.63 | 1250.56 | 0.00 | ORB-long ORB[1245.33,1250.10] vol=1.6x ATR=1.94 |
| Stop hit — per-position SL triggered | 2024-08-29 11:50:00 | 1249.69 | 1251.60 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-08-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 10:45:00 | 1250.33 | 1252.70 | 0.00 | ORB-short ORB[1251.60,1258.65] vol=1.6x ATR=2.51 |
| Stop hit — per-position SL triggered | 2024-08-30 11:35:00 | 1252.84 | 1252.43 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-09-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 11:10:00 | 1270.03 | 1263.10 | 0.00 | ORB-long ORB[1251.00,1259.38] vol=1.6x ATR=2.31 |
| Stop hit — per-position SL triggered | 2024-09-03 11:50:00 | 1267.72 | 1265.30 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-09-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 11:05:00 | 1254.95 | 1256.67 | 0.00 | ORB-short ORB[1259.00,1267.50] vol=2.5x ATR=1.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 11:15:00 | 1252.02 | 1256.21 | 0.00 | T1 1.5R @ 1252.02 |
| Stop hit — per-position SL triggered | 2024-09-05 12:20:00 | 1254.95 | 1254.51 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 09:35:00 | 1267.00 | 1263.01 | 0.00 | ORB-long ORB[1257.00,1264.50] vol=2.1x ATR=2.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 09:40:00 | 1270.87 | 1264.84 | 0.00 | T1 1.5R @ 1270.87 |
| Stop hit — per-position SL triggered | 2024-09-10 09:55:00 | 1267.00 | 1266.18 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-09-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 11:15:00 | 1280.40 | 1275.28 | 0.00 | ORB-long ORB[1264.80,1277.47] vol=1.7x ATR=2.69 |
| Stop hit — per-position SL triggered | 2024-09-13 11:20:00 | 1277.71 | 1275.37 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-09-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 10:45:00 | 1270.75 | 1261.61 | 0.00 | ORB-long ORB[1250.00,1265.00] vol=1.6x ATR=2.94 |
| Stop hit — per-position SL triggered | 2024-09-16 10:50:00 | 1267.81 | 1261.79 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 11:15:00 | 1284.83 | 1279.99 | 0.00 | ORB-long ORB[1274.00,1284.75] vol=1.9x ATR=2.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 11:25:00 | 1288.61 | 1281.22 | 0.00 | T1 1.5R @ 1288.61 |
| Stop hit — per-position SL triggered | 2024-09-17 12:55:00 | 1284.83 | 1284.27 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-09-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 10:45:00 | 1287.00 | 1282.16 | 0.00 | ORB-long ORB[1273.00,1280.00] vol=1.8x ATR=2.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 11:35:00 | 1290.58 | 1284.61 | 0.00 | T1 1.5R @ 1290.58 |
| Stop hit — per-position SL triggered | 2024-09-18 12:05:00 | 1287.00 | 1285.34 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 10:15:00 | 1322.00 | 1307.90 | 0.00 | ORB-long ORB[1298.58,1306.33] vol=1.5x ATR=3.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:30:00 | 1327.96 | 1313.97 | 0.00 | T1 1.5R @ 1327.96 |
| Target hit | 2024-09-19 11:50:00 | 1323.15 | 1324.01 | 0.00 | Trail-exit close<VWAP |

### Cycle 39 — BUY (started 2024-09-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:50:00 | 1387.23 | 1378.81 | 0.00 | ORB-long ORB[1365.50,1376.23] vol=3.8x ATR=3.69 |
| Stop hit — per-position SL triggered | 2024-09-27 11:05:00 | 1383.54 | 1380.57 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-10-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 11:05:00 | 1297.97 | 1303.80 | 0.00 | ORB-short ORB[1298.85,1307.30] vol=1.9x ATR=4.24 |
| Stop hit — per-position SL triggered | 2024-10-07 11:20:00 | 1302.21 | 1303.07 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-10-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-09 09:45:00 | 1270.63 | 1278.03 | 0.00 | ORB-short ORB[1272.50,1290.00] vol=1.8x ATR=4.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-09 09:50:00 | 1263.98 | 1274.99 | 0.00 | T1 1.5R @ 1263.98 |
| Target hit | 2024-10-09 10:45:00 | 1254.47 | 1253.07 | 0.00 | Trail-exit close>VWAP |

### Cycle 42 — SELL (started 2024-10-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 10:55:00 | 1252.50 | 1252.91 | 0.00 | ORB-short ORB[1252.70,1260.00] vol=1.5x ATR=2.08 |
| Stop hit — per-position SL triggered | 2024-10-11 11:25:00 | 1254.58 | 1252.83 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-10-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-22 09:40:00 | 1190.72 | 1184.89 | 0.00 | ORB-long ORB[1173.53,1186.75] vol=1.8x ATR=3.19 |
| Stop hit — per-position SL triggered | 2024-10-22 09:45:00 | 1187.53 | 1185.62 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-10-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-28 10:50:00 | 1150.00 | 1139.69 | 0.00 | ORB-long ORB[1128.03,1139.00] vol=2.0x ATR=3.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 11:25:00 | 1154.92 | 1142.63 | 0.00 | T1 1.5R @ 1154.92 |
| Stop hit — per-position SL triggered | 2024-10-28 14:00:00 | 1150.00 | 1147.31 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-10-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 10:55:00 | 1133.70 | 1135.67 | 0.00 | ORB-short ORB[1134.50,1142.90] vol=3.3x ATR=2.97 |
| Stop hit — per-position SL triggered | 2024-10-29 12:00:00 | 1136.67 | 1134.74 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-11-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-04 10:25:00 | 1122.53 | 1131.67 | 0.00 | ORB-short ORB[1132.33,1143.50] vol=1.7x ATR=2.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-04 11:00:00 | 1118.15 | 1126.52 | 0.00 | T1 1.5R @ 1118.15 |
| Stop hit — per-position SL triggered | 2024-11-04 11:35:00 | 1122.53 | 1124.47 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-11-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-08 11:05:00 | 1141.28 | 1136.14 | 0.00 | ORB-long ORB[1129.40,1136.40] vol=2.7x ATR=2.27 |
| Stop hit — per-position SL triggered | 2024-11-08 11:10:00 | 1139.01 | 1136.46 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-11-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-14 11:05:00 | 1108.00 | 1111.64 | 0.00 | ORB-short ORB[1114.53,1122.47] vol=2.2x ATR=2.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 11:55:00 | 1104.07 | 1109.11 | 0.00 | T1 1.5R @ 1104.07 |
| Target hit | 2024-11-14 15:20:00 | 1090.75 | 1101.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — BUY (started 2024-11-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 11:00:00 | 1117.00 | 1112.25 | 0.00 | ORB-long ORB[1106.15,1115.00] vol=2.2x ATR=2.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 12:00:00 | 1120.63 | 1113.96 | 0.00 | T1 1.5R @ 1120.63 |
| Stop hit — per-position SL triggered | 2024-11-19 12:05:00 | 1117.00 | 1114.18 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-11-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 10:55:00 | 1112.53 | 1111.45 | 0.00 | ORB-long ORB[1102.10,1112.00] vol=6.2x ATR=3.64 |
| Stop hit — per-position SL triggered | 2024-11-22 11:10:00 | 1108.89 | 1111.25 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-11-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-26 11:00:00 | 1137.20 | 1129.63 | 0.00 | ORB-long ORB[1126.00,1135.00] vol=2.5x ATR=2.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-26 11:30:00 | 1140.78 | 1130.79 | 0.00 | T1 1.5R @ 1140.78 |
| Stop hit — per-position SL triggered | 2024-11-26 12:10:00 | 1137.20 | 1133.75 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-11-28 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:25:00 | 1132.65 | 1135.12 | 0.00 | ORB-short ORB[1133.33,1139.97] vol=2.0x ATR=2.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 10:30:00 | 1129.44 | 1134.75 | 0.00 | T1 1.5R @ 1129.44 |
| Stop hit — per-position SL triggered | 2024-11-28 15:05:00 | 1132.65 | 1125.57 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-11-29 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 10:30:00 | 1126.78 | 1122.28 | 0.00 | ORB-long ORB[1113.50,1123.83] vol=1.6x ATR=3.64 |
| Stop hit — per-position SL triggered | 2024-11-29 12:35:00 | 1123.14 | 1124.96 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-12-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:45:00 | 1116.10 | 1119.70 | 0.00 | ORB-short ORB[1123.00,1129.95] vol=2.1x ATR=2.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-05 10:55:00 | 1112.91 | 1118.70 | 0.00 | T1 1.5R @ 1112.91 |
| Stop hit — per-position SL triggered | 2024-12-05 12:05:00 | 1116.10 | 1117.63 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-12-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 11:10:00 | 1116.80 | 1118.45 | 0.00 | ORB-short ORB[1119.63,1127.00] vol=2.1x ATR=1.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 11:40:00 | 1114.14 | 1117.85 | 0.00 | T1 1.5R @ 1114.14 |
| Stop hit — per-position SL triggered | 2024-12-12 11:45:00 | 1116.80 | 1117.20 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-12-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 11:10:00 | 1110.25 | 1111.44 | 0.00 | ORB-short ORB[1112.50,1118.30] vol=2.4x ATR=2.39 |
| Stop hit — per-position SL triggered | 2024-12-13 11:20:00 | 1112.64 | 1111.45 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-12-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 11:00:00 | 1117.60 | 1118.82 | 0.00 | ORB-short ORB[1118.60,1128.75] vol=3.7x ATR=1.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 12:00:00 | 1114.98 | 1117.97 | 0.00 | T1 1.5R @ 1114.98 |
| Stop hit — per-position SL triggered | 2024-12-16 14:50:00 | 1117.60 | 1117.10 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-12-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 10:35:00 | 1106.70 | 1109.39 | 0.00 | ORB-short ORB[1110.50,1116.97] vol=1.9x ATR=2.04 |
| Stop hit — per-position SL triggered | 2024-12-17 10:55:00 | 1108.74 | 1109.05 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-12-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-18 10:50:00 | 1098.58 | 1101.48 | 0.00 | ORB-short ORB[1101.47,1106.97] vol=1.8x ATR=1.93 |
| Stop hit — per-position SL triggered | 2024-12-18 11:25:00 | 1100.51 | 1099.25 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-12-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-19 10:50:00 | 1080.88 | 1084.03 | 0.00 | ORB-short ORB[1084.50,1097.13] vol=1.7x ATR=2.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 11:00:00 | 1077.21 | 1083.59 | 0.00 | T1 1.5R @ 1077.21 |
| Target hit | 2024-12-19 15:10:00 | 1079.50 | 1078.94 | 0.00 | Trail-exit close>VWAP |

### Cycle 61 — BUY (started 2024-12-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 10:30:00 | 1082.25 | 1081.45 | 0.00 | ORB-long ORB[1072.70,1080.00] vol=2.6x ATR=3.06 |
| Stop hit — per-position SL triggered | 2024-12-20 13:05:00 | 1079.19 | 1082.80 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2024-12-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 11:05:00 | 1086.05 | 1081.63 | 0.00 | ORB-long ORB[1075.85,1083.00] vol=2.0x ATR=1.81 |
| Stop hit — per-position SL triggered | 2024-12-24 12:15:00 | 1084.24 | 1083.38 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-01-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:55:00 | 1089.13 | 1086.28 | 0.00 | ORB-long ORB[1083.13,1089.00] vol=2.4x ATR=1.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 10:25:00 | 1092.04 | 1088.01 | 0.00 | T1 1.5R @ 1092.04 |
| Target hit | 2025-01-02 15:20:00 | 1100.25 | 1096.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — BUY (started 2025-01-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 11:00:00 | 1110.00 | 1106.82 | 0.00 | ORB-long ORB[1099.03,1109.97] vol=2.1x ATR=2.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 11:15:00 | 1113.66 | 1107.39 | 0.00 | T1 1.5R @ 1113.66 |
| Stop hit — per-position SL triggered | 2025-01-03 12:25:00 | 1110.00 | 1109.13 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-01-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 10:45:00 | 1100.28 | 1105.16 | 0.00 | ORB-short ORB[1112.50,1122.45] vol=3.0x ATR=3.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:10:00 | 1095.63 | 1103.75 | 0.00 | T1 1.5R @ 1095.63 |
| Target hit | 2025-01-06 15:20:00 | 1093.22 | 1095.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 66 — BUY (started 2025-01-09 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 10:10:00 | 1113.25 | 1107.98 | 0.00 | ORB-long ORB[1102.03,1109.95] vol=1.7x ATR=2.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 10:20:00 | 1117.50 | 1111.18 | 0.00 | T1 1.5R @ 1117.50 |
| Target hit | 2025-01-09 15:20:00 | 1126.63 | 1126.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 67 — BUY (started 2025-01-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-13 10:55:00 | 1127.63 | 1118.69 | 0.00 | ORB-long ORB[1111.85,1122.50] vol=1.6x ATR=3.01 |
| Stop hit — per-position SL triggered | 2025-01-13 11:10:00 | 1124.62 | 1119.54 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-01-14 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-14 10:35:00 | 1103.65 | 1110.55 | 0.00 | ORB-short ORB[1114.40,1128.30] vol=2.8x ATR=3.34 |
| Stop hit — per-position SL triggered | 2025-01-14 10:50:00 | 1106.99 | 1109.32 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-01-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 10:50:00 | 1090.43 | 1094.36 | 0.00 | ORB-short ORB[1095.58,1110.05] vol=3.9x ATR=2.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-15 11:00:00 | 1086.39 | 1092.72 | 0.00 | T1 1.5R @ 1086.39 |
| Stop hit — per-position SL triggered | 2025-01-15 11:10:00 | 1090.43 | 1092.50 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-01-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-20 10:50:00 | 1103.65 | 1104.49 | 0.00 | ORB-short ORB[1103.85,1112.50] vol=3.6x ATR=1.88 |
| Stop hit — per-position SL triggered | 2025-01-20 11:00:00 | 1105.53 | 1104.49 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-01-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 11:00:00 | 1106.63 | 1107.89 | 0.00 | ORB-short ORB[1109.00,1115.35] vol=2.5x ATR=2.72 |
| Stop hit — per-position SL triggered | 2025-01-21 11:45:00 | 1109.35 | 1109.54 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-01-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-24 11:10:00 | 1105.13 | 1102.31 | 0.00 | ORB-long ORB[1095.05,1103.80] vol=5.3x ATR=1.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 11:15:00 | 1108.08 | 1102.93 | 0.00 | T1 1.5R @ 1108.08 |
| Stop hit — per-position SL triggered | 2025-01-24 12:00:00 | 1105.13 | 1106.59 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2025-01-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-28 09:50:00 | 1082.00 | 1084.97 | 0.00 | ORB-short ORB[1082.50,1090.93] vol=1.7x ATR=3.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 10:20:00 | 1077.45 | 1082.79 | 0.00 | T1 1.5R @ 1077.45 |
| Stop hit — per-position SL triggered | 2025-01-28 11:00:00 | 1082.00 | 1080.44 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2025-02-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 11:05:00 | 1119.00 | 1121.74 | 0.00 | ORB-short ORB[1121.15,1134.50] vol=2.4x ATR=2.11 |
| Stop hit — per-position SL triggered | 2025-02-06 11:45:00 | 1121.11 | 1121.49 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2025-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-14 10:15:00 | 1105.13 | 1095.84 | 0.00 | ORB-long ORB[1085.03,1093.20] vol=1.6x ATR=2.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 10:25:00 | 1109.62 | 1098.39 | 0.00 | T1 1.5R @ 1109.62 |
| Stop hit — per-position SL triggered | 2025-02-14 10:40:00 | 1105.13 | 1102.78 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2025-02-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-18 11:10:00 | 1100.50 | 1102.94 | 0.00 | ORB-short ORB[1101.47,1111.00] vol=2.8x ATR=2.30 |
| Stop hit — per-position SL triggered | 2025-02-18 11:15:00 | 1102.80 | 1102.57 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2025-02-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-20 09:35:00 | 1096.80 | 1099.48 | 0.00 | ORB-short ORB[1097.50,1107.47] vol=2.1x ATR=2.54 |
| Stop hit — per-position SL triggered | 2025-02-20 10:30:00 | 1099.34 | 1098.33 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2025-02-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 09:50:00 | 1095.78 | 1097.33 | 0.00 | ORB-short ORB[1098.50,1108.75] vol=8.5x ATR=2.31 |
| Stop hit — per-position SL triggered | 2025-02-21 10:05:00 | 1098.09 | 1097.28 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2025-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-24 11:15:00 | 1113.53 | 1109.38 | 0.00 | ORB-long ORB[1098.03,1113.43] vol=1.7x ATR=2.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-24 11:20:00 | 1116.96 | 1109.84 | 0.00 | T1 1.5R @ 1116.96 |
| Stop hit — per-position SL triggered | 2025-02-24 13:20:00 | 1113.53 | 1112.22 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2025-02-25 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 10:20:00 | 1122.60 | 1118.11 | 0.00 | ORB-long ORB[1106.93,1119.45] vol=2.3x ATR=2.59 |
| Stop hit — per-position SL triggered | 2025-02-25 11:05:00 | 1120.01 | 1118.99 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2025-02-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-28 10:40:00 | 1123.40 | 1126.87 | 0.00 | ORB-short ORB[1126.60,1135.00] vol=3.7x ATR=2.88 |
| Stop hit — per-position SL triggered | 2025-02-28 11:20:00 | 1126.28 | 1125.46 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2025-03-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 11:05:00 | 1118.63 | 1106.61 | 0.00 | ORB-long ORB[1095.63,1106.50] vol=2.2x ATR=2.86 |
| Stop hit — per-position SL triggered | 2025-03-07 11:35:00 | 1115.77 | 1108.54 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2025-03-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-17 09:35:00 | 1087.05 | 1089.96 | 0.00 | ORB-short ORB[1087.78,1101.68] vol=2.0x ATR=2.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-17 09:45:00 | 1082.71 | 1088.71 | 0.00 | T1 1.5R @ 1082.71 |
| Target hit | 2025-03-17 11:05:00 | 1085.70 | 1085.32 | 0.00 | Trail-exit close>VWAP |

### Cycle 84 — BUY (started 2025-03-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:00:00 | 1095.60 | 1091.71 | 0.00 | ORB-long ORB[1085.00,1094.43] vol=1.8x ATR=2.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 10:20:00 | 1099.34 | 1092.95 | 0.00 | T1 1.5R @ 1099.34 |
| Target hit | 2025-03-18 12:40:00 | 1096.38 | 1096.62 | 0.00 | Trail-exit close<VWAP |

### Cycle 85 — BUY (started 2025-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 10:15:00 | 1104.10 | 1100.38 | 0.00 | ORB-long ORB[1090.58,1102.00] vol=4.3x ATR=2.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-20 13:15:00 | 1107.96 | 1102.39 | 0.00 | T1 1.5R @ 1107.96 |
| Target hit | 2025-03-20 15:20:00 | 1110.47 | 1105.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 86 — BUY (started 2025-03-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 11:00:00 | 1131.00 | 1125.13 | 0.00 | ORB-long ORB[1115.97,1129.75] vol=2.6x ATR=2.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 11:15:00 | 1134.84 | 1126.49 | 0.00 | T1 1.5R @ 1134.84 |
| Stop hit — per-position SL triggered | 2025-03-21 13:20:00 | 1131.00 | 1129.34 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2025-03-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-25 09:50:00 | 1139.88 | 1132.31 | 0.00 | ORB-long ORB[1123.75,1137.78] vol=1.9x ATR=3.17 |
| Stop hit — per-position SL triggered | 2025-03-25 09:55:00 | 1136.71 | 1133.52 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2025-03-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 10:30:00 | 1132.00 | 1125.01 | 0.00 | ORB-long ORB[1112.63,1125.00] vol=5.3x ATR=3.14 |
| Stop hit — per-position SL triggered | 2025-03-27 11:15:00 | 1128.86 | 1127.54 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2025-04-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-04 09:30:00 | 1127.22 | 1117.05 | 0.00 | ORB-long ORB[1108.63,1123.88] vol=1.5x ATR=3.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 10:15:00 | 1132.73 | 1125.72 | 0.00 | T1 1.5R @ 1132.73 |
| Stop hit — per-position SL triggered | 2025-04-04 10:35:00 | 1127.22 | 1126.79 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2025-04-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-11 10:40:00 | 1164.50 | 1173.89 | 0.00 | ORB-short ORB[1172.85,1186.00] vol=2.0x ATR=3.70 |
| Stop hit — per-position SL triggered | 2025-04-11 11:35:00 | 1168.20 | 1172.25 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2025-04-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 11:00:00 | 1189.95 | 1186.64 | 0.00 | ORB-long ORB[1179.30,1187.50] vol=2.0x ATR=2.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-17 11:10:00 | 1193.86 | 1188.10 | 0.00 | T1 1.5R @ 1193.86 |
| Target hit | 2025-04-17 15:20:00 | 1208.00 | 1202.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 92 — BUY (started 2025-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 09:30:00 | 1220.00 | 1212.63 | 0.00 | ORB-long ORB[1200.00,1216.90] vol=1.6x ATR=3.46 |
| Stop hit — per-position SL triggered | 2025-04-23 09:35:00 | 1216.54 | 1213.53 | 0.00 | SL hit |

### Cycle 93 — BUY (started 2025-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 09:30:00 | 1237.70 | 1226.97 | 0.00 | ORB-long ORB[1212.90,1229.40] vol=2.5x ATR=4.06 |
| Stop hit — per-position SL triggered | 2025-04-24 09:35:00 | 1233.64 | 1227.82 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2025-04-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 09:35:00 | 1188.55 | 1193.85 | 0.00 | ORB-short ORB[1193.85,1203.00] vol=2.0x ATR=3.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 09:40:00 | 1183.81 | 1191.33 | 0.00 | T1 1.5R @ 1183.81 |
| Stop hit — per-position SL triggered | 2025-04-29 10:10:00 | 1188.55 | 1188.21 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2025-05-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-06 09:55:00 | 1180.00 | 1175.29 | 0.00 | ORB-long ORB[1165.00,1172.50] vol=1.6x ATR=2.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 10:55:00 | 1184.04 | 1178.79 | 0.00 | T1 1.5R @ 1184.04 |
| Stop hit — per-position SL triggered | 2025-05-06 11:20:00 | 1180.00 | 1179.57 | 0.00 | SL hit |

### Cycle 96 — BUY (started 2025-05-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 10:45:00 | 1169.10 | 1164.63 | 0.00 | ORB-long ORB[1157.55,1167.35] vol=1.6x ATR=2.32 |
| Stop hit — per-position SL triggered | 2025-05-08 11:00:00 | 1166.78 | 1165.00 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-14 09:30:00 | 1244.43 | 2024-05-14 09:45:00 | 1247.60 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-05-16 09:30:00 | 1226.30 | 2024-05-16 10:20:00 | 1222.40 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-05-16 09:30:00 | 1226.30 | 2024-05-16 12:05:00 | 1223.15 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2024-05-18 09:45:00 | 1257.97 | 2024-05-18 09:50:00 | 1254.06 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-05-29 10:45:00 | 1235.85 | 2024-05-29 11:10:00 | 1239.87 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-05-29 10:45:00 | 1235.85 | 2024-05-29 11:20:00 | 1235.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-07 10:35:00 | 1245.35 | 2024-06-07 11:35:00 | 1250.89 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-06-07 10:35:00 | 1245.35 | 2024-06-07 13:00:00 | 1245.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-10 11:15:00 | 1275.00 | 2024-06-10 11:45:00 | 1279.84 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-06-10 11:15:00 | 1275.00 | 2024-06-10 14:35:00 | 1275.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-12 10:05:00 | 1263.22 | 2024-06-12 10:15:00 | 1266.51 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-06-14 11:15:00 | 1271.95 | 2024-06-14 11:30:00 | 1268.60 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2024-06-14 11:15:00 | 1271.95 | 2024-06-14 13:05:00 | 1271.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-18 11:05:00 | 1271.00 | 2024-06-18 12:30:00 | 1267.72 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2024-06-18 11:05:00 | 1271.00 | 2024-06-18 13:10:00 | 1271.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-21 10:40:00 | 1254.43 | 2024-06-21 11:35:00 | 1257.44 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-06-26 10:55:00 | 1265.38 | 2024-06-26 12:00:00 | 1268.67 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2024-06-26 10:55:00 | 1265.38 | 2024-06-26 15:00:00 | 1265.38 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-28 11:15:00 | 1284.65 | 2024-06-28 11:45:00 | 1281.74 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-07-01 10:40:00 | 1293.03 | 2024-07-01 11:25:00 | 1289.87 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-07-03 11:15:00 | 1281.10 | 2024-07-03 11:25:00 | 1278.72 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-07-08 10:00:00 | 1300.03 | 2024-07-08 10:10:00 | 1305.79 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-07-08 10:00:00 | 1300.03 | 2024-07-08 10:30:00 | 1300.03 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-11 09:35:00 | 1296.18 | 2024-07-11 10:10:00 | 1289.47 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-07-11 09:35:00 | 1296.18 | 2024-07-11 12:15:00 | 1293.33 | TARGET_HIT | 0.50 | 0.22% |
| BUY | retest1 | 2024-07-12 10:55:00 | 1305.50 | 2024-07-12 11:10:00 | 1301.77 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-07-16 10:20:00 | 1289.25 | 2024-07-16 10:40:00 | 1284.65 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-07-16 10:20:00 | 1289.25 | 2024-07-16 10:45:00 | 1289.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-22 10:55:00 | 1295.00 | 2024-07-22 12:05:00 | 1289.71 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-07-22 10:55:00 | 1295.00 | 2024-07-22 13:25:00 | 1295.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-25 09:40:00 | 1268.72 | 2024-07-25 10:15:00 | 1272.52 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-07-30 10:20:00 | 1227.95 | 2024-07-30 10:25:00 | 1230.01 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2024-08-01 10:30:00 | 1242.50 | 2024-08-01 10:35:00 | 1246.12 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2024-08-01 10:30:00 | 1242.50 | 2024-08-01 10:40:00 | 1242.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-02 11:00:00 | 1236.03 | 2024-08-02 11:10:00 | 1239.70 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-08-05 10:00:00 | 1262.45 | 2024-08-05 10:10:00 | 1258.30 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-08-08 10:55:00 | 1246.20 | 2024-08-08 11:15:00 | 1248.68 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-08-21 10:05:00 | 1272.55 | 2024-08-21 10:35:00 | 1269.87 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-08-27 10:35:00 | 1258.10 | 2024-08-27 10:45:00 | 1260.44 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-08-28 10:45:00 | 1250.00 | 2024-08-28 11:05:00 | 1247.01 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2024-08-28 10:45:00 | 1250.00 | 2024-08-28 11:15:00 | 1250.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-29 10:55:00 | 1251.63 | 2024-08-29 11:50:00 | 1249.69 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2024-08-30 10:45:00 | 1250.33 | 2024-08-30 11:35:00 | 1252.84 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-09-03 11:10:00 | 1270.03 | 2024-09-03 11:50:00 | 1267.72 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-09-05 11:05:00 | 1254.95 | 2024-09-05 11:15:00 | 1252.02 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2024-09-05 11:05:00 | 1254.95 | 2024-09-05 12:20:00 | 1254.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-10 09:35:00 | 1267.00 | 2024-09-10 09:40:00 | 1270.87 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2024-09-10 09:35:00 | 1267.00 | 2024-09-10 09:55:00 | 1267.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-13 11:15:00 | 1280.40 | 2024-09-13 11:20:00 | 1277.71 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-09-16 10:45:00 | 1270.75 | 2024-09-16 10:50:00 | 1267.81 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-09-17 11:15:00 | 1284.83 | 2024-09-17 11:25:00 | 1288.61 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2024-09-17 11:15:00 | 1284.83 | 2024-09-17 12:55:00 | 1284.83 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-18 10:45:00 | 1287.00 | 2024-09-18 11:35:00 | 1290.58 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2024-09-18 10:45:00 | 1287.00 | 2024-09-18 12:05:00 | 1287.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-19 10:15:00 | 1322.00 | 2024-09-19 10:30:00 | 1327.96 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-09-19 10:15:00 | 1322.00 | 2024-09-19 11:50:00 | 1323.15 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2024-09-27 10:50:00 | 1387.23 | 2024-09-27 11:05:00 | 1383.54 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-10-07 11:05:00 | 1297.97 | 2024-10-07 11:20:00 | 1302.21 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-10-09 09:45:00 | 1270.63 | 2024-10-09 09:50:00 | 1263.98 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-10-09 09:45:00 | 1270.63 | 2024-10-09 10:45:00 | 1254.47 | TARGET_HIT | 0.50 | 1.27% |
| SELL | retest1 | 2024-10-11 10:55:00 | 1252.50 | 2024-10-11 11:25:00 | 1254.58 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2024-10-22 09:40:00 | 1190.72 | 2024-10-22 09:45:00 | 1187.53 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-10-28 10:50:00 | 1150.00 | 2024-10-28 11:25:00 | 1154.92 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-10-28 10:50:00 | 1150.00 | 2024-10-28 14:00:00 | 1150.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-29 10:55:00 | 1133.70 | 2024-10-29 12:00:00 | 1136.67 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-11-04 10:25:00 | 1122.53 | 2024-11-04 11:00:00 | 1118.15 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-11-04 10:25:00 | 1122.53 | 2024-11-04 11:35:00 | 1122.53 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-08 11:05:00 | 1141.28 | 2024-11-08 11:10:00 | 1139.01 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-11-14 11:05:00 | 1108.00 | 2024-11-14 11:55:00 | 1104.07 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-11-14 11:05:00 | 1108.00 | 2024-11-14 15:20:00 | 1090.75 | TARGET_HIT | 0.50 | 1.56% |
| BUY | retest1 | 2024-11-19 11:00:00 | 1117.00 | 2024-11-19 12:00:00 | 1120.63 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-11-19 11:00:00 | 1117.00 | 2024-11-19 12:05:00 | 1117.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-22 10:55:00 | 1112.53 | 2024-11-22 11:10:00 | 1108.89 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-11-26 11:00:00 | 1137.20 | 2024-11-26 11:30:00 | 1140.78 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-11-26 11:00:00 | 1137.20 | 2024-11-26 12:10:00 | 1137.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-28 10:25:00 | 1132.65 | 2024-11-28 10:30:00 | 1129.44 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2024-11-28 10:25:00 | 1132.65 | 2024-11-28 15:05:00 | 1132.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-29 10:30:00 | 1126.78 | 2024-11-29 12:35:00 | 1123.14 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-12-05 10:45:00 | 1116.10 | 2024-12-05 10:55:00 | 1112.91 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-12-05 10:45:00 | 1116.10 | 2024-12-05 12:05:00 | 1116.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-12 11:10:00 | 1116.80 | 2024-12-12 11:40:00 | 1114.14 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2024-12-12 11:10:00 | 1116.80 | 2024-12-12 11:45:00 | 1116.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-13 11:10:00 | 1110.25 | 2024-12-13 11:20:00 | 1112.64 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-12-16 11:00:00 | 1117.60 | 2024-12-16 12:00:00 | 1114.98 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2024-12-16 11:00:00 | 1117.60 | 2024-12-16 14:50:00 | 1117.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-17 10:35:00 | 1106.70 | 2024-12-17 10:55:00 | 1108.74 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-12-18 10:50:00 | 1098.58 | 2024-12-18 11:25:00 | 1100.51 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-12-19 10:50:00 | 1080.88 | 2024-12-19 11:00:00 | 1077.21 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-12-19 10:50:00 | 1080.88 | 2024-12-19 15:10:00 | 1079.50 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2024-12-20 10:30:00 | 1082.25 | 2024-12-20 13:05:00 | 1079.19 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-12-24 11:05:00 | 1086.05 | 2024-12-24 12:15:00 | 1084.24 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-01-02 09:55:00 | 1089.13 | 2025-01-02 10:25:00 | 1092.04 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2025-01-02 09:55:00 | 1089.13 | 2025-01-02 15:20:00 | 1100.25 | TARGET_HIT | 0.50 | 1.02% |
| BUY | retest1 | 2025-01-03 11:00:00 | 1110.00 | 2025-01-03 11:15:00 | 1113.66 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-01-03 11:00:00 | 1110.00 | 2025-01-03 12:25:00 | 1110.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-06 10:45:00 | 1100.28 | 2025-01-06 11:10:00 | 1095.63 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-01-06 10:45:00 | 1100.28 | 2025-01-06 15:20:00 | 1093.22 | TARGET_HIT | 0.50 | 0.64% |
| BUY | retest1 | 2025-01-09 10:10:00 | 1113.25 | 2025-01-09 10:20:00 | 1117.50 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-01-09 10:10:00 | 1113.25 | 2025-01-09 15:20:00 | 1126.63 | TARGET_HIT | 0.50 | 1.20% |
| BUY | retest1 | 2025-01-13 10:55:00 | 1127.63 | 2025-01-13 11:10:00 | 1124.62 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-01-14 10:35:00 | 1103.65 | 2025-01-14 10:50:00 | 1106.99 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-01-15 10:50:00 | 1090.43 | 2025-01-15 11:00:00 | 1086.39 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-01-15 10:50:00 | 1090.43 | 2025-01-15 11:10:00 | 1090.43 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-20 10:50:00 | 1103.65 | 2025-01-20 11:00:00 | 1105.53 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-01-21 11:00:00 | 1106.63 | 2025-01-21 11:45:00 | 1109.35 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-01-24 11:10:00 | 1105.13 | 2025-01-24 11:15:00 | 1108.08 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2025-01-24 11:10:00 | 1105.13 | 2025-01-24 12:00:00 | 1105.13 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-28 09:50:00 | 1082.00 | 2025-01-28 10:20:00 | 1077.45 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-01-28 09:50:00 | 1082.00 | 2025-01-28 11:00:00 | 1082.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-06 11:05:00 | 1119.00 | 2025-02-06 11:45:00 | 1121.11 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-02-14 10:15:00 | 1105.13 | 2025-02-14 10:25:00 | 1109.62 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-02-14 10:15:00 | 1105.13 | 2025-02-14 10:40:00 | 1105.13 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-18 11:10:00 | 1100.50 | 2025-02-18 11:15:00 | 1102.80 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-02-20 09:35:00 | 1096.80 | 2025-02-20 10:30:00 | 1099.34 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-02-21 09:50:00 | 1095.78 | 2025-02-21 10:05:00 | 1098.09 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-02-24 11:15:00 | 1113.53 | 2025-02-24 11:20:00 | 1116.96 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-02-24 11:15:00 | 1113.53 | 2025-02-24 13:20:00 | 1113.53 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-25 10:20:00 | 1122.60 | 2025-02-25 11:05:00 | 1120.01 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-02-28 10:40:00 | 1123.40 | 2025-02-28 11:20:00 | 1126.28 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-03-07 11:05:00 | 1118.63 | 2025-03-07 11:35:00 | 1115.77 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-03-17 09:35:00 | 1087.05 | 2025-03-17 09:45:00 | 1082.71 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-03-17 09:35:00 | 1087.05 | 2025-03-17 11:05:00 | 1085.70 | TARGET_HIT | 0.50 | 0.12% |
| BUY | retest1 | 2025-03-18 10:00:00 | 1095.60 | 2025-03-18 10:20:00 | 1099.34 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-03-18 10:00:00 | 1095.60 | 2025-03-18 12:40:00 | 1096.38 | TARGET_HIT | 0.50 | 0.07% |
| BUY | retest1 | 2025-03-20 10:15:00 | 1104.10 | 2025-03-20 13:15:00 | 1107.96 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-03-20 10:15:00 | 1104.10 | 2025-03-20 15:20:00 | 1110.47 | TARGET_HIT | 0.50 | 0.58% |
| BUY | retest1 | 2025-03-21 11:00:00 | 1131.00 | 2025-03-21 11:15:00 | 1134.84 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-03-21 11:00:00 | 1131.00 | 2025-03-21 13:20:00 | 1131.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-25 09:50:00 | 1139.88 | 2025-03-25 09:55:00 | 1136.71 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-03-27 10:30:00 | 1132.00 | 2025-03-27 11:15:00 | 1128.86 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-04-04 09:30:00 | 1127.22 | 2025-04-04 10:15:00 | 1132.73 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-04-04 09:30:00 | 1127.22 | 2025-04-04 10:35:00 | 1127.22 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-11 10:40:00 | 1164.50 | 2025-04-11 11:35:00 | 1168.20 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-04-17 11:00:00 | 1189.95 | 2025-04-17 11:10:00 | 1193.86 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-04-17 11:00:00 | 1189.95 | 2025-04-17 15:20:00 | 1208.00 | TARGET_HIT | 0.50 | 1.52% |
| BUY | retest1 | 2025-04-23 09:30:00 | 1220.00 | 2025-04-23 09:35:00 | 1216.54 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-04-24 09:30:00 | 1237.70 | 2025-04-24 09:35:00 | 1233.64 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-04-29 09:35:00 | 1188.55 | 2025-04-29 09:40:00 | 1183.81 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-04-29 09:35:00 | 1188.55 | 2025-04-29 10:10:00 | 1188.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-06 09:55:00 | 1180.00 | 2025-05-06 10:55:00 | 1184.04 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-05-06 09:55:00 | 1180.00 | 2025-05-06 11:20:00 | 1180.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-08 10:45:00 | 1169.10 | 2025-05-08 11:00:00 | 1166.78 | STOP_HIT | 1.00 | -0.20% |
