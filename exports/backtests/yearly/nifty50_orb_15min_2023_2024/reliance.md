# RELIANCE (RELIANCE)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (55350 bars)
- **Last close:** 1436.00
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
| ENTRY1 | 84 |
| ENTRY2 | 0 |
| PARTIAL | 27 |
| TARGET_HIT | 11 |
| STOP_HIT | 73 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 111 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 38 / 73
- **Target hits / Stop hits / Partials:** 11 / 73 / 27
- **Avg / median % per leg:** 0.04% / -0.12%
- **Sum % (uncompounded):** 4.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 63 | 18 | 28.6% | 4 | 45 | 14 | -0.01% | -0.4% |
| BUY @ 2nd Alert (retest1) | 63 | 18 | 28.6% | 4 | 45 | 14 | -0.01% | -0.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 48 | 20 | 41.7% | 7 | 28 | 13 | 0.10% | 4.7% |
| SELL @ 2nd Alert (retest1) | 48 | 20 | 41.7% | 7 | 28 | 13 | 0.10% | 4.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 111 | 38 | 34.2% | 11 | 73 | 27 | 0.04% | 4.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-15 11:10:00 | 1245.25 | 1241.13 | 0.00 | ORB-long ORB[1236.58,1245.00] vol=1.9x ATR=1.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-15 11:50:00 | 1247.51 | 1242.23 | 0.00 | T1 1.5R @ 1247.51 |
| Stop hit — per-position SL triggered | 2023-05-15 15:00:00 | 1245.25 | 1246.44 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-18 11:00:00 | 1222.35 | 1223.75 | 0.00 | ORB-short ORB[1224.00,1228.50] vol=1.7x ATR=2.30 |
| Stop hit — per-position SL triggered | 2023-05-18 11:15:00 | 1224.65 | 1223.75 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-05-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-22 10:55:00 | 1229.50 | 1227.04 | 0.00 | ORB-long ORB[1216.18,1227.45] vol=2.4x ATR=1.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-22 11:10:00 | 1232.26 | 1227.66 | 0.00 | T1 1.5R @ 1232.26 |
| Stop hit — per-position SL triggered | 2023-05-22 11:45:00 | 1229.50 | 1228.39 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-30 10:10:00 | 1267.55 | 1263.67 | 0.00 | ORB-long ORB[1257.18,1264.50] vol=2.3x ATR=1.84 |
| Stop hit — per-position SL triggered | 2023-05-30 10:30:00 | 1265.71 | 1264.43 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-06-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-01 11:15:00 | 1242.00 | 1237.04 | 0.00 | ORB-long ORB[1231.18,1240.43] vol=2.8x ATR=2.24 |
| Stop hit — per-position SL triggered | 2023-06-01 11:20:00 | 1239.76 | 1237.19 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2023-06-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-02 11:05:00 | 1227.50 | 1231.52 | 0.00 | ORB-short ORB[1234.50,1241.43] vol=6.0x ATR=1.81 |
| Stop hit — per-position SL triggered | 2023-06-02 11:55:00 | 1229.31 | 1230.08 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-06 11:15:00 | 1233.88 | 1238.90 | 0.00 | ORB-short ORB[1237.75,1243.85] vol=1.8x ATR=1.80 |
| Stop hit — per-position SL triggered | 2023-06-06 11:25:00 | 1235.68 | 1238.51 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-06-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-14 09:30:00 | 1264.18 | 1262.32 | 0.00 | ORB-long ORB[1258.50,1263.50] vol=1.7x ATR=1.94 |
| Stop hit — per-position SL triggered | 2023-06-14 09:35:00 | 1262.24 | 1262.32 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-06-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-15 11:10:00 | 1276.50 | 1275.34 | 0.00 | ORB-long ORB[1269.80,1276.00] vol=2.6x ATR=1.95 |
| Stop hit — per-position SL triggered | 2023-06-15 11:20:00 | 1274.55 | 1275.35 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2023-06-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-19 11:10:00 | 1274.60 | 1280.89 | 0.00 | ORB-short ORB[1281.70,1292.00] vol=2.7x ATR=1.98 |
| Stop hit — per-position SL triggered | 2023-06-19 12:15:00 | 1276.58 | 1279.46 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2023-06-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-22 11:15:00 | 1277.10 | 1282.60 | 0.00 | ORB-short ORB[1277.53,1286.08] vol=1.7x ATR=1.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-22 11:40:00 | 1274.28 | 1281.46 | 0.00 | T1 1.5R @ 1274.28 |
| Target hit | 2023-06-22 15:20:00 | 1267.35 | 1275.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2023-06-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-27 10:30:00 | 1247.00 | 1249.75 | 0.00 | ORB-short ORB[1248.28,1254.75] vol=3.0x ATR=1.77 |
| Stop hit — per-position SL triggered | 2023-06-27 11:50:00 | 1248.77 | 1248.99 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-06-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-28 10:45:00 | 1261.25 | 1255.96 | 0.00 | ORB-long ORB[1252.50,1256.50] vol=1.7x ATR=1.68 |
| Stop hit — per-position SL triggered | 2023-06-28 10:50:00 | 1259.57 | 1256.11 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2023-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-04 11:15:00 | 1292.33 | 1294.15 | 0.00 | ORB-short ORB[1294.88,1312.50] vol=1.6x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-04 11:45:00 | 1289.08 | 1293.55 | 0.00 | T1 1.5R @ 1289.08 |
| Stop hit — per-position SL triggered | 2023-07-04 11:55:00 | 1292.33 | 1293.39 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-18 09:30:00 | 1400.05 | 1404.08 | 0.00 | ORB-short ORB[1400.30,1413.35] vol=1.6x ATR=3.49 |
| Stop hit — per-position SL triggered | 2023-07-18 09:45:00 | 1403.54 | 1403.39 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-07-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-19 09:30:00 | 1418.08 | 1415.25 | 0.00 | ORB-long ORB[1410.00,1417.50] vol=2.2x ATR=3.00 |
| Stop hit — per-position SL triggered | 2023-07-19 09:35:00 | 1415.08 | 1415.38 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-07-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 10:40:00 | 1250.70 | 1246.04 | 0.00 | ORB-long ORB[1240.00,1247.50] vol=2.6x ATR=2.50 |
| Stop hit — per-position SL triggered | 2023-07-25 10:55:00 | 1248.20 | 1246.25 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-07-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 09:40:00 | 1251.75 | 1247.91 | 0.00 | ORB-long ORB[1242.50,1250.00] vol=1.8x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-26 09:45:00 | 1255.40 | 1249.59 | 0.00 | T1 1.5R @ 1255.40 |
| Target hit | 2023-07-26 13:50:00 | 1263.03 | 1263.27 | 0.00 | Trail-exit close<VWAP |

### Cycle 19 — SELL (started 2023-07-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-27 09:50:00 | 1260.83 | 1263.93 | 0.00 | ORB-short ORB[1261.58,1268.83] vol=1.6x ATR=2.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-27 10:40:00 | 1256.98 | 1261.90 | 0.00 | T1 1.5R @ 1256.98 |
| Stop hit — per-position SL triggered | 2023-07-27 11:55:00 | 1260.83 | 1260.69 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-07-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-28 09:30:00 | 1264.72 | 1255.37 | 0.00 | ORB-long ORB[1250.28,1257.00] vol=1.6x ATR=3.61 |
| Stop hit — per-position SL triggered | 2023-07-28 09:35:00 | 1261.11 | 1257.36 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-08-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-07 11:10:00 | 1256.50 | 1259.16 | 0.00 | ORB-short ORB[1256.70,1264.20] vol=1.6x ATR=1.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-07 11:30:00 | 1253.78 | 1258.76 | 0.00 | T1 1.5R @ 1253.78 |
| Stop hit — per-position SL triggered | 2023-08-07 12:20:00 | 1256.50 | 1257.88 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-08-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-08 11:00:00 | 1257.33 | 1262.99 | 0.00 | ORB-short ORB[1259.05,1265.50] vol=2.6x ATR=2.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-08 13:00:00 | 1254.19 | 1260.69 | 0.00 | T1 1.5R @ 1254.19 |
| Stop hit — per-position SL triggered | 2023-08-08 13:20:00 | 1257.33 | 1260.42 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-08-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-09 10:55:00 | 1241.30 | 1245.14 | 0.00 | ORB-short ORB[1245.10,1253.45] vol=2.0x ATR=1.75 |
| Stop hit — per-position SL triggered | 2023-08-09 11:25:00 | 1243.05 | 1244.18 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-08-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-18 10:50:00 | 1257.63 | 1259.14 | 0.00 | ORB-short ORB[1260.85,1267.03] vol=2.1x ATR=2.62 |
| Stop hit — per-position SL triggered | 2023-08-18 11:15:00 | 1260.25 | 1259.13 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-08-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-23 10:30:00 | 1266.75 | 1264.23 | 0.00 | ORB-long ORB[1260.30,1266.50] vol=2.0x ATR=1.95 |
| Stop hit — per-position SL triggered | 2023-08-23 10:45:00 | 1264.80 | 1264.59 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-08-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-24 10:55:00 | 1259.08 | 1265.01 | 0.00 | ORB-short ORB[1262.60,1269.95] vol=1.9x ATR=2.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-24 11:25:00 | 1256.04 | 1263.61 | 0.00 | T1 1.5R @ 1256.04 |
| Target hit | 2023-08-24 15:20:00 | 1237.43 | 1252.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2023-08-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-25 09:55:00 | 1250.22 | 1238.52 | 0.00 | ORB-long ORB[1221.30,1237.03] vol=2.1x ATR=4.42 |
| Stop hit — per-position SL triggered | 2023-08-25 10:05:00 | 1245.80 | 1240.01 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2023-08-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-29 11:05:00 | 1210.13 | 1215.51 | 0.00 | ORB-short ORB[1211.88,1226.72] vol=2.4x ATR=2.48 |
| Stop hit — per-position SL triggered | 2023-08-29 12:15:00 | 1212.61 | 1214.61 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-08-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-30 09:40:00 | 1210.70 | 1212.24 | 0.00 | ORB-short ORB[1210.80,1216.00] vol=1.6x ATR=1.95 |
| Stop hit — per-position SL triggered | 2023-08-30 09:50:00 | 1212.65 | 1212.31 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-08-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-31 11:00:00 | 1206.53 | 1208.87 | 0.00 | ORB-short ORB[1207.53,1212.50] vol=1.9x ATR=1.62 |
| Stop hit — per-position SL triggered | 2023-08-31 11:30:00 | 1208.15 | 1208.42 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2023-09-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 10:05:00 | 1211.90 | 1210.13 | 0.00 | ORB-long ORB[1206.22,1210.50] vol=1.9x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-05 10:20:00 | 1214.41 | 1210.95 | 0.00 | T1 1.5R @ 1214.41 |
| Stop hit — per-position SL triggered | 2023-09-05 10:30:00 | 1211.90 | 1211.23 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-09-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 09:35:00 | 1217.25 | 1215.26 | 0.00 | ORB-long ORB[1210.55,1216.68] vol=1.7x ATR=1.82 |
| Stop hit — per-position SL triggered | 2023-09-06 09:50:00 | 1215.43 | 1215.60 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-09-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-27 09:55:00 | 1175.05 | 1173.28 | 0.00 | ORB-long ORB[1169.25,1175.00] vol=2.1x ATR=1.43 |
| Stop hit — per-position SL triggered | 2023-09-27 10:00:00 | 1173.62 | 1173.34 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-10-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-09 09:50:00 | 1155.30 | 1151.85 | 0.00 | ORB-long ORB[1148.55,1154.75] vol=2.0x ATR=2.10 |
| Stop hit — per-position SL triggered | 2023-10-09 10:35:00 | 1153.20 | 1152.78 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-10-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-18 09:30:00 | 1179.30 | 1177.03 | 0.00 | ORB-long ORB[1174.50,1178.75] vol=1.8x ATR=1.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-18 09:35:00 | 1181.83 | 1178.19 | 0.00 | T1 1.5R @ 1181.83 |
| Stop hit — per-position SL triggered | 2023-10-18 10:40:00 | 1179.30 | 1180.14 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-10-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-27 10:55:00 | 1130.83 | 1128.17 | 0.00 | ORB-long ORB[1117.97,1126.40] vol=1.5x ATR=2.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-27 12:00:00 | 1134.54 | 1128.95 | 0.00 | T1 1.5R @ 1134.54 |
| Stop hit — per-position SL triggered | 2023-10-27 12:15:00 | 1130.83 | 1129.13 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-10-31 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-31 10:55:00 | 1152.50 | 1156.25 | 0.00 | ORB-short ORB[1154.50,1164.00] vol=4.1x ATR=2.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-31 11:40:00 | 1149.36 | 1155.36 | 0.00 | T1 1.5R @ 1149.36 |
| Target hit | 2023-10-31 15:20:00 | 1142.78 | 1151.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — BUY (started 2023-11-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-01 11:00:00 | 1155.00 | 1148.64 | 0.00 | ORB-long ORB[1137.60,1148.22] vol=1.8x ATR=2.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-01 12:15:00 | 1158.21 | 1150.74 | 0.00 | T1 1.5R @ 1158.21 |
| Stop hit — per-position SL triggered | 2023-11-01 12:35:00 | 1155.00 | 1151.08 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-11-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 10:00:00 | 1161.40 | 1158.34 | 0.00 | ORB-long ORB[1153.97,1159.95] vol=2.1x ATR=2.24 |
| Stop hit — per-position SL triggered | 2023-11-02 10:10:00 | 1159.16 | 1158.81 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2023-11-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-09 10:25:00 | 1160.53 | 1162.33 | 0.00 | ORB-short ORB[1161.75,1167.93] vol=1.6x ATR=1.41 |
| Stop hit — per-position SL triggered | 2023-11-09 11:05:00 | 1161.94 | 1161.66 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-11-15 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-15 10:25:00 | 1172.50 | 1169.69 | 0.00 | ORB-long ORB[1163.50,1170.00] vol=5.4x ATR=1.82 |
| Stop hit — per-position SL triggered | 2023-11-15 11:30:00 | 1170.68 | 1170.18 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2023-11-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 11:00:00 | 1183.00 | 1178.64 | 0.00 | ORB-long ORB[1173.65,1177.20] vol=2.3x ATR=1.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-16 14:05:00 | 1185.69 | 1181.97 | 0.00 | T1 1.5R @ 1185.69 |
| Stop hit — per-position SL triggered | 2023-11-16 15:10:00 | 1183.00 | 1182.44 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2023-11-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-20 11:10:00 | 1169.22 | 1173.77 | 0.00 | ORB-short ORB[1174.28,1179.20] vol=1.8x ATR=1.41 |
| Stop hit — per-position SL triggered | 2023-11-20 11:30:00 | 1170.63 | 1173.35 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-11-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 09:40:00 | 1188.22 | 1183.29 | 0.00 | ORB-long ORB[1180.10,1185.08] vol=1.7x ATR=2.09 |
| Stop hit — per-position SL triggered | 2023-11-21 09:55:00 | 1186.13 | 1184.39 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2023-11-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-22 09:30:00 | 1196.70 | 1193.99 | 0.00 | ORB-long ORB[1187.50,1196.18] vol=1.9x ATR=1.86 |
| Stop hit — per-position SL triggered | 2023-11-22 09:55:00 | 1194.84 | 1194.45 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2023-11-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-23 09:30:00 | 1198.33 | 1196.98 | 0.00 | ORB-long ORB[1194.10,1197.95] vol=2.7x ATR=1.61 |
| Stop hit — per-position SL triggered | 2023-11-23 10:30:00 | 1196.72 | 1198.09 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2023-11-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-28 09:35:00 | 1190.63 | 1192.28 | 0.00 | ORB-short ORB[1191.00,1196.95] vol=4.9x ATR=1.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-28 11:00:00 | 1188.23 | 1191.01 | 0.00 | T1 1.5R @ 1188.23 |
| Target hit | 2023-11-28 14:15:00 | 1190.00 | 1189.94 | 0.00 | Trail-exit close>VWAP |

### Cycle 48 — SELL (started 2023-12-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-04 10:35:00 | 1201.60 | 1210.42 | 0.00 | ORB-short ORB[1207.15,1225.00] vol=2.9x ATR=2.75 |
| Stop hit — per-position SL triggered | 2023-12-04 10:55:00 | 1204.35 | 1207.38 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2023-12-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-05 11:10:00 | 1219.85 | 1217.05 | 0.00 | ORB-long ORB[1211.00,1219.50] vol=2.4x ATR=1.58 |
| Stop hit — per-position SL triggered | 2023-12-05 11:15:00 | 1218.27 | 1217.11 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2023-12-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-06 09:40:00 | 1233.58 | 1225.45 | 0.00 | ORB-long ORB[1217.50,1227.28] vol=1.6x ATR=2.32 |
| Stop hit — per-position SL triggered | 2023-12-06 09:55:00 | 1231.26 | 1226.81 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2023-12-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-14 09:30:00 | 1233.18 | 1225.74 | 0.00 | ORB-long ORB[1221.33,1227.50] vol=2.4x ATR=3.08 |
| Stop hit — per-position SL triggered | 2023-12-14 09:35:00 | 1230.10 | 1226.41 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2023-12-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-19 10:30:00 | 1266.83 | 1268.95 | 0.00 | ORB-short ORB[1267.50,1277.50] vol=2.4x ATR=2.87 |
| Stop hit — per-position SL triggered | 2023-12-19 10:40:00 | 1269.70 | 1268.97 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2023-12-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-20 10:55:00 | 1297.60 | 1292.45 | 0.00 | ORB-long ORB[1285.53,1297.50] vol=2.8x ATR=2.64 |
| Stop hit — per-position SL triggered | 2023-12-20 11:30:00 | 1294.96 | 1293.61 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-01-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-02 09:35:00 | 1302.85 | 1297.72 | 0.00 | ORB-long ORB[1291.55,1299.70] vol=1.5x ATR=2.67 |
| Stop hit — per-position SL triggered | 2024-01-02 09:50:00 | 1300.18 | 1298.58 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-01-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 09:35:00 | 1297.80 | 1293.90 | 0.00 | ORB-long ORB[1289.55,1294.28] vol=2.3x ATR=2.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-04 10:00:00 | 1301.26 | 1296.04 | 0.00 | T1 1.5R @ 1301.26 |
| Stop hit — per-position SL triggered | 2024-01-04 11:15:00 | 1297.80 | 1297.91 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-01-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 10:35:00 | 1309.45 | 1304.27 | 0.00 | ORB-long ORB[1300.00,1306.93] vol=2.0x ATR=2.09 |
| Stop hit — per-position SL triggered | 2024-01-05 10:40:00 | 1307.36 | 1304.41 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2024-01-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 09:35:00 | 1343.48 | 1338.35 | 0.00 | ORB-long ORB[1328.50,1342.03] vol=1.6x ATR=4.05 |
| Stop hit — per-position SL triggered | 2024-01-11 10:00:00 | 1339.43 | 1339.21 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2024-01-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-12 11:00:00 | 1364.78 | 1355.90 | 0.00 | ORB-long ORB[1345.75,1360.58] vol=2.6x ATR=3.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-12 12:15:00 | 1369.38 | 1358.87 | 0.00 | T1 1.5R @ 1369.38 |
| Target hit | 2024-01-12 15:20:00 | 1372.40 | 1364.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — BUY (started 2024-01-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-16 10:45:00 | 1395.80 | 1389.49 | 0.00 | ORB-long ORB[1383.43,1391.58] vol=2.9x ATR=3.11 |
| Stop hit — per-position SL triggered | 2024-01-16 10:50:00 | 1392.69 | 1389.74 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2024-02-01 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-01 10:10:00 | 1439.48 | 1430.62 | 0.00 | ORB-long ORB[1426.63,1435.00] vol=1.5x ATR=4.40 |
| Stop hit — per-position SL triggered | 2024-02-01 10:15:00 | 1435.08 | 1431.24 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2024-02-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-06 10:30:00 | 1423.15 | 1431.13 | 0.00 | ORB-short ORB[1426.90,1441.85] vol=2.6x ATR=5.05 |
| Stop hit — per-position SL triggered | 2024-02-06 10:45:00 | 1428.20 | 1429.80 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-02-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 10:15:00 | 1438.13 | 1442.66 | 0.00 | ORB-short ORB[1442.18,1450.00] vol=1.6x ATR=3.17 |
| Stop hit — per-position SL triggered | 2024-02-08 10:25:00 | 1441.30 | 1442.47 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2024-02-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-12 10:55:00 | 1447.78 | 1448.81 | 0.00 | ORB-short ORB[1448.23,1461.00] vol=1.8x ATR=3.56 |
| Stop hit — per-position SL triggered | 2024-02-12 11:10:00 | 1451.34 | 1448.87 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2024-02-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-13 11:05:00 | 1469.38 | 1461.22 | 0.00 | ORB-long ORB[1454.00,1463.33] vol=3.5x ATR=3.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-13 11:10:00 | 1473.90 | 1464.15 | 0.00 | T1 1.5R @ 1473.90 |
| Stop hit — per-position SL triggered | 2024-02-13 12:35:00 | 1469.38 | 1468.15 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2024-02-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-21 11:00:00 | 1484.98 | 1476.69 | 0.00 | ORB-long ORB[1471.05,1478.50] vol=2.7x ATR=3.25 |
| Stop hit — per-position SL triggered | 2024-02-21 11:15:00 | 1481.73 | 1477.51 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2024-02-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-26 10:50:00 | 1484.00 | 1488.15 | 0.00 | ORB-short ORB[1487.70,1494.53] vol=5.1x ATR=2.35 |
| Stop hit — per-position SL triggered | 2024-02-26 11:05:00 | 1486.35 | 1488.02 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2024-02-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 10:45:00 | 1479.23 | 1485.37 | 0.00 | ORB-short ORB[1480.50,1489.73] vol=2.2x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-28 10:55:00 | 1475.58 | 1484.32 | 0.00 | T1 1.5R @ 1475.58 |
| Target hit | 2024-02-28 15:20:00 | 1457.28 | 1465.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 68 — BUY (started 2024-03-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-04 10:30:00 | 1498.00 | 1494.24 | 0.00 | ORB-long ORB[1487.23,1496.50] vol=2.0x ATR=2.69 |
| Stop hit — per-position SL triggered | 2024-03-04 10:45:00 | 1495.31 | 1494.51 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2024-03-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-05 10:20:00 | 1491.83 | 1495.97 | 0.00 | ORB-short ORB[1494.73,1507.40] vol=2.6x ATR=2.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-05 10:30:00 | 1487.91 | 1494.54 | 0.00 | T1 1.5R @ 1487.91 |
| Stop hit — per-position SL triggered | 2024-03-05 10:40:00 | 1491.83 | 1494.17 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2024-03-07 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-07 10:25:00 | 1491.90 | 1496.83 | 0.00 | ORB-short ORB[1495.03,1502.98] vol=2.0x ATR=3.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-07 10:40:00 | 1487.34 | 1495.71 | 0.00 | T1 1.5R @ 1487.34 |
| Stop hit — per-position SL triggered | 2024-03-07 11:00:00 | 1491.90 | 1494.74 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2024-03-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-12 11:10:00 | 1480.60 | 1479.16 | 0.00 | ORB-long ORB[1465.03,1474.50] vol=2.0x ATR=3.45 |
| Stop hit — per-position SL triggered | 2024-03-12 11:40:00 | 1477.15 | 1479.22 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2024-03-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-14 11:15:00 | 1426.33 | 1439.60 | 0.00 | ORB-short ORB[1434.00,1448.53] vol=3.1x ATR=3.87 |
| Stop hit — per-position SL triggered | 2024-03-14 11:30:00 | 1430.20 | 1438.20 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2024-03-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-20 10:10:00 | 1430.50 | 1430.33 | 0.00 | ORB-long ORB[1424.03,1430.25] vol=2.6x ATR=3.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 11:10:00 | 1435.25 | 1430.91 | 0.00 | T1 1.5R @ 1435.25 |
| Target hit | 2024-03-20 15:20:00 | 1444.93 | 1437.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 74 — BUY (started 2024-03-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 10:20:00 | 1456.10 | 1450.75 | 0.00 | ORB-long ORB[1445.70,1454.50] vol=1.6x ATR=2.99 |
| Stop hit — per-position SL triggered | 2024-03-21 11:40:00 | 1453.11 | 1452.36 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2024-03-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 09:45:00 | 1500.90 | 1495.89 | 0.00 | ORB-long ORB[1490.03,1498.48] vol=1.5x ATR=3.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-28 10:00:00 | 1505.73 | 1498.11 | 0.00 | T1 1.5R @ 1505.73 |
| Stop hit — per-position SL triggered | 2024-03-28 10:25:00 | 1500.90 | 1499.40 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-04-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-01 11:10:00 | 1487.75 | 1489.69 | 0.00 | ORB-short ORB[1487.88,1493.98] vol=1.8x ATR=2.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-01 13:05:00 | 1484.06 | 1488.43 | 0.00 | T1 1.5R @ 1484.06 |
| Target hit | 2024-04-01 15:20:00 | 1484.28 | 1487.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 77 — SELL (started 2024-04-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 10:50:00 | 1450.73 | 1465.05 | 0.00 | ORB-short ORB[1469.00,1479.75] vol=2.2x ATR=3.51 |
| Stop hit — per-position SL triggered | 2024-04-04 10:55:00 | 1454.24 | 1464.49 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2024-04-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-08 09:50:00 | 1483.70 | 1476.57 | 0.00 | ORB-long ORB[1461.00,1476.28] vol=1.6x ATR=3.41 |
| Stop hit — per-position SL triggered | 2024-04-08 10:25:00 | 1480.29 | 1479.25 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2024-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-10 09:30:00 | 1485.25 | 1477.65 | 0.00 | ORB-long ORB[1466.35,1482.45] vol=1.6x ATR=3.41 |
| Stop hit — per-position SL triggered | 2024-04-10 09:40:00 | 1481.84 | 1478.15 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2024-04-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-12 09:35:00 | 1484.00 | 1478.34 | 0.00 | ORB-long ORB[1468.00,1480.50] vol=2.0x ATR=3.06 |
| Stop hit — per-position SL triggered | 2024-04-12 09:40:00 | 1480.94 | 1478.63 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2024-04-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 09:30:00 | 1467.00 | 1460.65 | 0.00 | ORB-long ORB[1450.00,1465.10] vol=1.8x ATR=3.53 |
| Stop hit — per-position SL triggered | 2024-04-29 09:45:00 | 1463.47 | 1461.59 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2024-04-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 09:45:00 | 1472.45 | 1469.05 | 0.00 | ORB-long ORB[1465.30,1471.50] vol=1.6x ATR=2.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-30 09:55:00 | 1476.35 | 1470.81 | 0.00 | T1 1.5R @ 1476.35 |
| Target hit | 2024-04-30 14:30:00 | 1476.68 | 1477.07 | 0.00 | Trail-exit close<VWAP |

### Cycle 83 — SELL (started 2024-05-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-06 11:00:00 | 1426.08 | 1431.91 | 0.00 | ORB-short ORB[1433.48,1440.00] vol=1.6x ATR=2.93 |
| Stop hit — per-position SL triggered | 2024-05-06 11:40:00 | 1429.01 | 1431.05 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2024-05-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-07 10:05:00 | 1410.20 | 1414.83 | 0.00 | ORB-short ORB[1414.08,1420.75] vol=1.7x ATR=2.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 10:15:00 | 1406.31 | 1413.30 | 0.00 | T1 1.5R @ 1406.31 |
| Target hit | 2024-05-07 15:20:00 | 1402.38 | 1402.29 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-15 11:10:00 | 1245.25 | 2023-05-15 11:50:00 | 1247.51 | PARTIAL | 0.50 | 0.18% |
| BUY | retest1 | 2023-05-15 11:10:00 | 1245.25 | 2023-05-15 15:00:00 | 1245.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-05-18 11:00:00 | 1222.35 | 2023-05-18 11:15:00 | 1224.65 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-05-22 10:55:00 | 1229.50 | 2023-05-22 11:10:00 | 1232.26 | PARTIAL | 0.50 | 0.22% |
| BUY | retest1 | 2023-05-22 10:55:00 | 1229.50 | 2023-05-22 11:45:00 | 1229.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-30 10:10:00 | 1267.55 | 2023-05-30 10:30:00 | 1265.71 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2023-06-01 11:15:00 | 1242.00 | 2023-06-01 11:20:00 | 1239.76 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-06-02 11:05:00 | 1227.50 | 2023-06-02 11:55:00 | 1229.31 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2023-06-06 11:15:00 | 1233.88 | 2023-06-06 11:25:00 | 1235.68 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2023-06-14 09:30:00 | 1264.18 | 2023-06-14 09:35:00 | 1262.24 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2023-06-15 11:10:00 | 1276.50 | 2023-06-15 11:20:00 | 1274.55 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2023-06-19 11:10:00 | 1274.60 | 2023-06-19 12:15:00 | 1276.58 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-06-22 11:15:00 | 1277.10 | 2023-06-22 11:40:00 | 1274.28 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2023-06-22 11:15:00 | 1277.10 | 2023-06-22 15:20:00 | 1267.35 | TARGET_HIT | 0.50 | 0.76% |
| SELL | retest1 | 2023-06-27 10:30:00 | 1247.00 | 2023-06-27 11:50:00 | 1248.77 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2023-06-28 10:45:00 | 1261.25 | 2023-06-28 10:50:00 | 1259.57 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2023-07-04 11:15:00 | 1292.33 | 2023-07-04 11:45:00 | 1289.08 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2023-07-04 11:15:00 | 1292.33 | 2023-07-04 11:55:00 | 1292.33 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-18 09:30:00 | 1400.05 | 2023-07-18 09:45:00 | 1403.54 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-07-19 09:30:00 | 1418.08 | 2023-07-19 09:35:00 | 1415.08 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-07-25 10:40:00 | 1250.70 | 2023-07-25 10:55:00 | 1248.20 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-07-26 09:40:00 | 1251.75 | 2023-07-26 09:45:00 | 1255.40 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2023-07-26 09:40:00 | 1251.75 | 2023-07-26 13:50:00 | 1263.03 | TARGET_HIT | 0.50 | 0.90% |
| SELL | retest1 | 2023-07-27 09:50:00 | 1260.83 | 2023-07-27 10:40:00 | 1256.98 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2023-07-27 09:50:00 | 1260.83 | 2023-07-27 11:55:00 | 1260.83 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-28 09:30:00 | 1264.72 | 2023-07-28 09:35:00 | 1261.11 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-08-07 11:10:00 | 1256.50 | 2023-08-07 11:30:00 | 1253.78 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2023-08-07 11:10:00 | 1256.50 | 2023-08-07 12:20:00 | 1256.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-08 11:00:00 | 1257.33 | 2023-08-08 13:00:00 | 1254.19 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2023-08-08 11:00:00 | 1257.33 | 2023-08-08 13:20:00 | 1257.33 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-09 10:55:00 | 1241.30 | 2023-08-09 11:25:00 | 1243.05 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2023-08-18 10:50:00 | 1257.63 | 2023-08-18 11:15:00 | 1260.25 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-08-23 10:30:00 | 1266.75 | 2023-08-23 10:45:00 | 1264.80 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2023-08-24 10:55:00 | 1259.08 | 2023-08-24 11:25:00 | 1256.04 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2023-08-24 10:55:00 | 1259.08 | 2023-08-24 15:20:00 | 1237.43 | TARGET_HIT | 0.50 | 1.72% |
| BUY | retest1 | 2023-08-25 09:55:00 | 1250.22 | 2023-08-25 10:05:00 | 1245.80 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2023-08-29 11:05:00 | 1210.13 | 2023-08-29 12:15:00 | 1212.61 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-08-30 09:40:00 | 1210.70 | 2023-08-30 09:50:00 | 1212.65 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-08-31 11:00:00 | 1206.53 | 2023-08-31 11:30:00 | 1208.15 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2023-09-05 10:05:00 | 1211.90 | 2023-09-05 10:20:00 | 1214.41 | PARTIAL | 0.50 | 0.21% |
| BUY | retest1 | 2023-09-05 10:05:00 | 1211.90 | 2023-09-05 10:30:00 | 1211.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-06 09:35:00 | 1217.25 | 2023-09-06 09:50:00 | 1215.43 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2023-09-27 09:55:00 | 1175.05 | 2023-09-27 10:00:00 | 1173.62 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest1 | 2023-10-09 09:50:00 | 1155.30 | 2023-10-09 10:35:00 | 1153.20 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-10-18 09:30:00 | 1179.30 | 2023-10-18 09:35:00 | 1181.83 | PARTIAL | 0.50 | 0.21% |
| BUY | retest1 | 2023-10-18 09:30:00 | 1179.30 | 2023-10-18 10:40:00 | 1179.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-27 10:55:00 | 1130.83 | 2023-10-27 12:00:00 | 1134.54 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-10-27 10:55:00 | 1130.83 | 2023-10-27 12:15:00 | 1130.83 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-31 10:55:00 | 1152.50 | 2023-10-31 11:40:00 | 1149.36 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2023-10-31 10:55:00 | 1152.50 | 2023-10-31 15:20:00 | 1142.78 | TARGET_HIT | 0.50 | 0.84% |
| BUY | retest1 | 2023-11-01 11:00:00 | 1155.00 | 2023-11-01 12:15:00 | 1158.21 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2023-11-01 11:00:00 | 1155.00 | 2023-11-01 12:35:00 | 1155.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-02 10:00:00 | 1161.40 | 2023-11-02 10:10:00 | 1159.16 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-11-09 10:25:00 | 1160.53 | 2023-11-09 11:05:00 | 1161.94 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest1 | 2023-11-15 10:25:00 | 1172.50 | 2023-11-15 11:30:00 | 1170.68 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-11-16 11:00:00 | 1183.00 | 2023-11-16 14:05:00 | 1185.69 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2023-11-16 11:00:00 | 1183.00 | 2023-11-16 15:10:00 | 1183.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-20 11:10:00 | 1169.22 | 2023-11-20 11:30:00 | 1170.63 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest1 | 2023-11-21 09:40:00 | 1188.22 | 2023-11-21 09:55:00 | 1186.13 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-11-22 09:30:00 | 1196.70 | 2023-11-22 09:55:00 | 1194.84 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-11-23 09:30:00 | 1198.33 | 2023-11-23 10:30:00 | 1196.72 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2023-11-28 09:35:00 | 1190.63 | 2023-11-28 11:00:00 | 1188.23 | PARTIAL | 0.50 | 0.20% |
| SELL | retest1 | 2023-11-28 09:35:00 | 1190.63 | 2023-11-28 14:15:00 | 1190.00 | TARGET_HIT | 0.50 | 0.05% |
| SELL | retest1 | 2023-12-04 10:35:00 | 1201.60 | 2023-12-04 10:55:00 | 1204.35 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-12-05 11:10:00 | 1219.85 | 2023-12-05 11:15:00 | 1218.27 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2023-12-06 09:40:00 | 1233.58 | 2023-12-06 09:55:00 | 1231.26 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-12-14 09:30:00 | 1233.18 | 2023-12-14 09:35:00 | 1230.10 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-12-19 10:30:00 | 1266.83 | 2023-12-19 10:40:00 | 1269.70 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-12-20 10:55:00 | 1297.60 | 2023-12-20 11:30:00 | 1294.96 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-01-02 09:35:00 | 1302.85 | 2024-01-02 09:50:00 | 1300.18 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-01-04 09:35:00 | 1297.80 | 2024-01-04 10:00:00 | 1301.26 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2024-01-04 09:35:00 | 1297.80 | 2024-01-04 11:15:00 | 1297.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-05 10:35:00 | 1309.45 | 2024-01-05 10:40:00 | 1307.36 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2024-01-11 09:35:00 | 1343.48 | 2024-01-11 10:00:00 | 1339.43 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-01-12 11:00:00 | 1364.78 | 2024-01-12 12:15:00 | 1369.38 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-01-12 11:00:00 | 1364.78 | 2024-01-12 15:20:00 | 1372.40 | TARGET_HIT | 0.50 | 0.56% |
| BUY | retest1 | 2024-01-16 10:45:00 | 1395.80 | 2024-01-16 10:50:00 | 1392.69 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-02-01 10:10:00 | 1439.48 | 2024-02-01 10:15:00 | 1435.08 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-02-06 10:30:00 | 1423.15 | 2024-02-06 10:45:00 | 1428.20 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-02-08 10:15:00 | 1438.13 | 2024-02-08 10:25:00 | 1441.30 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-02-12 10:55:00 | 1447.78 | 2024-02-12 11:10:00 | 1451.34 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-02-13 11:05:00 | 1469.38 | 2024-02-13 11:10:00 | 1473.90 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2024-02-13 11:05:00 | 1469.38 | 2024-02-13 12:35:00 | 1469.38 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-21 11:00:00 | 1484.98 | 2024-02-21 11:15:00 | 1481.73 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-02-26 10:50:00 | 1484.00 | 2024-02-26 11:05:00 | 1486.35 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2024-02-28 10:45:00 | 1479.23 | 2024-02-28 10:55:00 | 1475.58 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2024-02-28 10:45:00 | 1479.23 | 2024-02-28 15:20:00 | 1457.28 | TARGET_HIT | 0.50 | 1.48% |
| BUY | retest1 | 2024-03-04 10:30:00 | 1498.00 | 2024-03-04 10:45:00 | 1495.31 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-03-05 10:20:00 | 1491.83 | 2024-03-05 10:30:00 | 1487.91 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2024-03-05 10:20:00 | 1491.83 | 2024-03-05 10:40:00 | 1491.83 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-07 10:25:00 | 1491.90 | 2024-03-07 10:40:00 | 1487.34 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-03-07 10:25:00 | 1491.90 | 2024-03-07 11:00:00 | 1491.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-12 11:10:00 | 1480.60 | 2024-03-12 11:40:00 | 1477.15 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-03-14 11:15:00 | 1426.33 | 2024-03-14 11:30:00 | 1430.20 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-03-20 10:10:00 | 1430.50 | 2024-03-20 11:10:00 | 1435.25 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-03-20 10:10:00 | 1430.50 | 2024-03-20 15:20:00 | 1444.93 | TARGET_HIT | 0.50 | 1.01% |
| BUY | retest1 | 2024-03-21 10:20:00 | 1456.10 | 2024-03-21 11:40:00 | 1453.11 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-03-28 09:45:00 | 1500.90 | 2024-03-28 10:00:00 | 1505.73 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-03-28 09:45:00 | 1500.90 | 2024-03-28 10:25:00 | 1500.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-01 11:10:00 | 1487.75 | 2024-04-01 13:05:00 | 1484.06 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2024-04-01 11:10:00 | 1487.75 | 2024-04-01 15:20:00 | 1484.28 | TARGET_HIT | 0.50 | 0.23% |
| SELL | retest1 | 2024-04-04 10:50:00 | 1450.73 | 2024-04-04 10:55:00 | 1454.24 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-04-08 09:50:00 | 1483.70 | 2024-04-08 10:25:00 | 1480.29 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-04-10 09:30:00 | 1485.25 | 2024-04-10 09:40:00 | 1481.84 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-04-12 09:35:00 | 1484.00 | 2024-04-12 09:40:00 | 1480.94 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-04-29 09:30:00 | 1467.00 | 2024-04-29 09:45:00 | 1463.47 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-04-30 09:45:00 | 1472.45 | 2024-04-30 09:55:00 | 1476.35 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2024-04-30 09:45:00 | 1472.45 | 2024-04-30 14:30:00 | 1476.68 | TARGET_HIT | 0.50 | 0.29% |
| SELL | retest1 | 2024-05-06 11:00:00 | 1426.08 | 2024-05-06 11:40:00 | 1429.01 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-05-07 10:05:00 | 1410.20 | 2024-05-07 10:15:00 | 1406.31 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2024-05-07 10:05:00 | 1410.20 | 2024-05-07 15:20:00 | 1402.38 | TARGET_HIT | 0.50 | 0.55% |
