# DRREDDY (DRREDDY)

## Backtest Summary

- **Window:** 2025-12-08 09:15:00 → 2026-05-08 15:25:00 (6375 bars)
- **Last close:** 1294.50
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
| ENTRY1 | 25 |
| ENTRY2 | 0 |
| PARTIAL | 9 |
| TARGET_HIT | 4 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 34 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 21
- **Target hits / Stop hits / Partials:** 4 / 21 / 9
- **Avg / median % per leg:** 0.04% / 0.00%
- **Sum % (uncompounded):** 1.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 9 | 45.0% | 3 | 11 | 6 | 0.10% | 2.1% |
| BUY @ 2nd Alert (retest1) | 20 | 9 | 45.0% | 3 | 11 | 6 | 0.10% | 2.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 4 | 28.6% | 1 | 10 | 3 | -0.05% | -0.7% |
| SELL @ 2nd Alert (retest1) | 14 | 4 | 28.6% | 1 | 10 | 3 | -0.05% | -0.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 34 | 13 | 38.2% | 4 | 21 | 9 | 0.04% | 1.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-12-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 10:20:00 | 1258.00 | 1254.45 | 0.00 | ORB-long ORB[1246.80,1251.90] vol=4.3x ATR=2.81 |
| Stop hit — per-position SL triggered | 2025-12-10 10:50:00 | 1255.19 | 1255.16 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-12-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 11:10:00 | 1261.40 | 1255.44 | 0.00 | ORB-long ORB[1248.00,1261.00] vol=5.5x ATR=2.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 11:30:00 | 1265.50 | 1255.90 | 0.00 | T1 1.5R @ 1265.50 |
| Target hit | 2025-12-11 15:20:00 | 1274.80 | 1263.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2025-12-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-12 10:50:00 | 1270.00 | 1271.69 | 0.00 | ORB-short ORB[1271.20,1278.50] vol=2.3x ATR=1.98 |
| Stop hit — per-position SL triggered | 2025-12-12 11:15:00 | 1271.98 | 1271.48 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-12-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 09:55:00 | 1265.30 | 1274.74 | 0.00 | ORB-short ORB[1275.00,1285.10] vol=1.6x ATR=3.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 10:00:00 | 1260.04 | 1270.17 | 0.00 | T1 1.5R @ 1260.04 |
| Stop hit — per-position SL triggered | 2025-12-24 12:00:00 | 1265.30 | 1264.02 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-12-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-31 11:10:00 | 1263.40 | 1264.24 | 0.00 | ORB-short ORB[1263.60,1270.00] vol=2.2x ATR=2.08 |
| Stop hit — per-position SL triggered | 2025-12-31 11:20:00 | 1265.48 | 1264.30 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-01-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 11:05:00 | 1250.50 | 1243.65 | 0.00 | ORB-long ORB[1236.00,1249.00] vol=1.9x ATR=2.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 11:15:00 | 1254.11 | 1243.93 | 0.00 | T1 1.5R @ 1254.11 |
| Stop hit — per-position SL triggered | 2026-01-02 11:30:00 | 1250.50 | 1244.42 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-01-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 10:20:00 | 1261.80 | 1259.11 | 0.00 | ORB-long ORB[1246.70,1256.90] vol=1.6x ATR=2.72 |
| Stop hit — per-position SL triggered | 2026-01-07 10:25:00 | 1259.08 | 1259.22 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-01-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 10:50:00 | 1196.30 | 1199.92 | 0.00 | ORB-short ORB[1198.10,1215.90] vol=2.7x ATR=3.68 |
| Stop hit — per-position SL triggered | 2026-01-13 11:00:00 | 1199.98 | 1199.66 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-01-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 10:35:00 | 1194.40 | 1190.51 | 0.00 | ORB-long ORB[1182.30,1194.00] vol=2.1x ATR=2.95 |
| Stop hit — per-position SL triggered | 2026-01-16 11:25:00 | 1191.45 | 1192.20 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-01-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-20 10:40:00 | 1176.40 | 1170.84 | 0.00 | ORB-long ORB[1164.60,1172.20] vol=8.4x ATR=3.37 |
| Stop hit — per-position SL triggered | 2026-01-20 10:50:00 | 1173.03 | 1172.19 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-01-28 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 10:25:00 | 1224.50 | 1232.20 | 0.00 | ORB-short ORB[1237.40,1243.20] vol=2.8x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-28 12:50:00 | 1220.19 | 1228.46 | 0.00 | T1 1.5R @ 1220.19 |
| Stop hit — per-position SL triggered | 2026-01-28 13:15:00 | 1224.50 | 1228.06 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-02-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 11:10:00 | 1228.40 | 1222.15 | 0.00 | ORB-long ORB[1217.10,1223.90] vol=8.3x ATR=3.88 |
| Stop hit — per-position SL triggered | 2026-02-01 11:25:00 | 1224.52 | 1222.81 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-02-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 10:25:00 | 1249.00 | 1243.35 | 0.00 | ORB-long ORB[1237.00,1245.90] vol=1.6x ATR=2.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-05 10:35:00 | 1252.39 | 1244.78 | 0.00 | T1 1.5R @ 1252.39 |
| Stop hit — per-position SL triggered | 2026-02-05 11:00:00 | 1249.00 | 1246.64 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-02-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 11:00:00 | 1230.30 | 1230.41 | 0.00 | ORB-short ORB[1231.40,1248.90] vol=10.1x ATR=2.81 |
| Stop hit — per-position SL triggered | 2026-02-06 11:05:00 | 1233.11 | 1230.49 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-02-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 09:30:00 | 1262.20 | 1256.11 | 0.00 | ORB-long ORB[1243.50,1260.80] vol=1.7x ATR=4.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 12:40:00 | 1268.30 | 1262.01 | 0.00 | T1 1.5R @ 1268.30 |
| Target hit | 2026-02-09 15:20:00 | 1272.40 | 1266.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2026-02-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 11:00:00 | 1266.40 | 1260.20 | 0.00 | ORB-long ORB[1253.20,1264.80] vol=1.9x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 11:40:00 | 1270.19 | 1262.34 | 0.00 | T1 1.5R @ 1270.19 |
| Stop hit — per-position SL triggered | 2026-02-11 12:35:00 | 1266.40 | 1264.12 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-02-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:55:00 | 1265.30 | 1269.59 | 0.00 | ORB-short ORB[1266.70,1274.00] vol=1.9x ATR=2.65 |
| Stop hit — per-position SL triggered | 2026-02-13 11:35:00 | 1267.95 | 1269.22 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-02-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:40:00 | 1280.50 | 1285.70 | 0.00 | ORB-short ORB[1282.20,1291.50] vol=1.7x ATR=3.25 |
| Stop hit — per-position SL triggered | 2026-02-18 09:45:00 | 1283.75 | 1285.52 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2026-02-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:45:00 | 1325.40 | 1316.16 | 0.00 | ORB-long ORB[1305.00,1314.90] vol=1.6x ATR=4.76 |
| Stop hit — per-position SL triggered | 2026-02-26 09:50:00 | 1320.64 | 1316.95 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2026-02-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:45:00 | 1291.30 | 1298.83 | 0.00 | ORB-short ORB[1303.00,1317.20] vol=1.9x ATR=3.77 |
| Stop hit — per-position SL triggered | 2026-02-27 11:00:00 | 1295.07 | 1298.16 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2026-03-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 09:45:00 | 1284.00 | 1281.72 | 0.00 | ORB-long ORB[1272.10,1283.50] vol=1.5x ATR=5.21 |
| Stop hit — per-position SL triggered | 2026-03-04 09:55:00 | 1278.79 | 1281.59 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 1294.80 | 1303.29 | 0.00 | ORB-short ORB[1304.80,1316.40] vol=1.7x ATR=3.33 |
| Stop hit — per-position SL triggered | 2026-03-06 10:50:00 | 1298.13 | 1303.05 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2026-04-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 11:10:00 | 1207.20 | 1204.27 | 0.00 | ORB-long ORB[1189.00,1205.00] vol=7.0x ATR=2.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 11:40:00 | 1211.43 | 1204.86 | 0.00 | T1 1.5R @ 1211.43 |
| Target hit | 2026-04-09 15:20:00 | 1212.80 | 1207.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — SELL (started 2026-04-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 10:55:00 | 1222.80 | 1225.63 | 0.00 | ORB-short ORB[1227.30,1234.90] vol=1.7x ATR=2.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 11:35:00 | 1219.71 | 1224.61 | 0.00 | T1 1.5R @ 1219.71 |
| Target hit | 2026-04-21 15:20:00 | 1220.70 | 1222.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — BUY (started 2026-05-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:40:00 | 1289.60 | 1283.70 | 0.00 | ORB-long ORB[1277.10,1285.30] vol=1.7x ATR=3.17 |
| Stop hit — per-position SL triggered | 2026-05-06 10:55:00 | 1286.43 | 1284.19 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-12-10 10:20:00 | 1258.00 | 2025-12-10 10:50:00 | 1255.19 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-12-11 11:10:00 | 1261.40 | 2025-12-11 11:30:00 | 1265.50 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-12-11 11:10:00 | 1261.40 | 2025-12-11 15:20:00 | 1274.80 | TARGET_HIT | 0.50 | 1.06% |
| SELL | retest1 | 2025-12-12 10:50:00 | 1270.00 | 2025-12-12 11:15:00 | 1271.98 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-12-24 09:55:00 | 1265.30 | 2025-12-24 10:00:00 | 1260.04 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-12-24 09:55:00 | 1265.30 | 2025-12-24 12:00:00 | 1265.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-31 11:10:00 | 1263.40 | 2025-12-31 11:20:00 | 1265.48 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2026-01-02 11:05:00 | 1250.50 | 2026-01-02 11:15:00 | 1254.11 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2026-01-02 11:05:00 | 1250.50 | 2026-01-02 11:30:00 | 1250.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-07 10:20:00 | 1261.80 | 2026-01-07 10:25:00 | 1259.08 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-01-13 10:50:00 | 1196.30 | 2026-01-13 11:00:00 | 1199.98 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-01-16 10:35:00 | 1194.40 | 2026-01-16 11:25:00 | 1191.45 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-01-20 10:40:00 | 1176.40 | 2026-01-20 10:50:00 | 1173.03 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-01-28 10:25:00 | 1224.50 | 2026-01-28 12:50:00 | 1220.19 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-01-28 10:25:00 | 1224.50 | 2026-01-28 13:15:00 | 1224.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-01 11:10:00 | 1228.40 | 2026-02-01 11:25:00 | 1224.52 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-05 10:25:00 | 1249.00 | 2026-02-05 10:35:00 | 1252.39 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2026-02-05 10:25:00 | 1249.00 | 2026-02-05 11:00:00 | 1249.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-06 11:00:00 | 1230.30 | 2026-02-06 11:05:00 | 1233.11 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-02-09 09:30:00 | 1262.20 | 2026-02-09 12:40:00 | 1268.30 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-02-09 09:30:00 | 1262.20 | 2026-02-09 15:20:00 | 1272.40 | TARGET_HIT | 0.50 | 0.81% |
| BUY | retest1 | 2026-02-11 11:00:00 | 1266.40 | 2026-02-11 11:40:00 | 1270.19 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-02-11 11:00:00 | 1266.40 | 2026-02-11 12:35:00 | 1266.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-13 10:55:00 | 1265.30 | 2026-02-13 11:35:00 | 1267.95 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-18 09:40:00 | 1280.50 | 2026-02-18 09:45:00 | 1283.75 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-26 09:45:00 | 1325.40 | 2026-02-26 09:50:00 | 1320.64 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-27 10:45:00 | 1291.30 | 2026-02-27 11:00:00 | 1295.07 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-03-04 09:45:00 | 1284.00 | 2026-03-04 09:55:00 | 1278.79 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-03-06 10:45:00 | 1294.80 | 2026-03-06 10:50:00 | 1298.13 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-04-09 11:10:00 | 1207.20 | 2026-04-09 11:40:00 | 1211.43 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-04-09 11:10:00 | 1207.20 | 2026-04-09 15:20:00 | 1212.80 | TARGET_HIT | 0.50 | 0.46% |
| SELL | retest1 | 2026-04-21 10:55:00 | 1222.80 | 2026-04-21 11:35:00 | 1219.71 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2026-04-21 10:55:00 | 1222.80 | 2026-04-21 15:20:00 | 1220.70 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2026-05-06 10:40:00 | 1289.60 | 2026-05-06 10:55:00 | 1286.43 | STOP_HIT | 1.00 | -0.25% |
