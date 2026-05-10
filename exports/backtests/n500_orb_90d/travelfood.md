# Travel Food Services Ltd. (TRAVELFOOD)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1250.00
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
| ENTRY1 | 18 |
| ENTRY2 | 0 |
| PARTIAL | 8 |
| TARGET_HIT | 2 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 16
- **Target hits / Stop hits / Partials:** 2 / 16 / 8
- **Avg / median % per leg:** 0.05% / 0.00%
- **Sum % (uncompounded):** 1.23%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 3 | 25.0% | 0 | 9 | 3 | -0.06% | -0.7% |
| BUY @ 2nd Alert (retest1) | 12 | 3 | 25.0% | 0 | 9 | 3 | -0.06% | -0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 7 | 50.0% | 2 | 7 | 5 | 0.14% | 2.0% |
| SELL @ 2nd Alert (retest1) | 14 | 7 | 50.0% | 2 | 7 | 5 | 0.14% | 2.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 26 | 10 | 38.5% | 2 | 16 | 8 | 0.05% | 1.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:25:00 | 1133.20 | 1119.37 | 0.00 | ORB-long ORB[1107.20,1112.10] vol=2.0x ATR=4.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:05:00 | 1140.28 | 1133.34 | 0.00 | T1 1.5R @ 1140.28 |
| Stop hit — per-position SL triggered | 2026-02-09 11:10:00 | 1133.20 | 1135.03 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:40:00 | 1177.60 | 1171.61 | 0.00 | ORB-long ORB[1160.00,1176.00] vol=3.4x ATR=7.44 |
| Stop hit — per-position SL triggered | 2026-02-10 10:05:00 | 1170.16 | 1171.79 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:50:00 | 1228.00 | 1221.04 | 0.00 | ORB-long ORB[1209.40,1223.00] vol=4.7x ATR=4.35 |
| Stop hit — per-position SL triggered | 2026-02-18 10:55:00 | 1223.65 | 1221.93 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:35:00 | 1208.00 | 1213.18 | 0.00 | ORB-short ORB[1212.00,1222.20] vol=2.7x ATR=3.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 09:40:00 | 1202.51 | 1210.98 | 0.00 | T1 1.5R @ 1202.51 |
| Stop hit — per-position SL triggered | 2026-02-19 10:25:00 | 1208.00 | 1207.87 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:50:00 | 1212.90 | 1221.28 | 0.00 | ORB-short ORB[1215.90,1233.50] vol=3.6x ATR=3.98 |
| Stop hit — per-position SL triggered | 2026-02-24 12:05:00 | 1216.88 | 1218.01 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 10:50:00 | 1172.00 | 1168.70 | 0.00 | ORB-long ORB[1150.20,1165.60] vol=3.2x ATR=3.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 14:20:00 | 1177.09 | 1175.23 | 0.00 | T1 1.5R @ 1177.09 |
| Stop hit — per-position SL triggered | 2026-03-16 15:10:00 | 1172.00 | 1175.01 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 1133.50 | 1129.13 | 0.00 | ORB-long ORB[1118.40,1130.50] vol=2.4x ATR=4.70 |
| Stop hit — per-position SL triggered | 2026-03-18 10:35:00 | 1128.80 | 1133.20 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-06 09:35:00 | 1290.00 | 1278.14 | 0.00 | ORB-long ORB[1266.30,1280.00] vol=2.8x ATR=5.16 |
| Stop hit — per-position SL triggered | 2026-04-06 09:40:00 | 1284.84 | 1278.95 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 11:00:00 | 1328.40 | 1319.13 | 0.00 | ORB-long ORB[1307.10,1323.70] vol=5.1x ATR=5.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 11:10:00 | 1336.29 | 1325.20 | 0.00 | T1 1.5R @ 1336.29 |
| Stop hit — per-position SL triggered | 2026-04-17 12:05:00 | 1328.40 | 1328.97 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 11:10:00 | 1308.00 | 1297.21 | 0.00 | ORB-long ORB[1288.60,1298.30] vol=3.5x ATR=3.94 |
| Stop hit — per-position SL triggered | 2026-04-21 11:20:00 | 1304.06 | 1297.60 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:50:00 | 1273.90 | 1279.20 | 0.00 | ORB-short ORB[1278.50,1290.00] vol=2.1x ATR=4.12 |
| Stop hit — per-position SL triggered | 2026-04-24 10:00:00 | 1278.02 | 1279.05 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 10:10:00 | 1288.80 | 1295.11 | 0.00 | ORB-short ORB[1292.00,1305.00] vol=1.8x ATR=3.82 |
| Stop hit — per-position SL triggered | 2026-04-27 10:15:00 | 1292.62 | 1303.10 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 11:10:00 | 1295.00 | 1286.23 | 0.00 | ORB-long ORB[1277.90,1294.90] vol=4.8x ATR=3.73 |
| Stop hit — per-position SL triggered | 2026-04-28 11:15:00 | 1291.27 | 1286.29 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 11:00:00 | 1279.00 | 1282.55 | 0.00 | ORB-short ORB[1283.00,1294.90] vol=5.9x ATR=1.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 12:45:00 | 1276.24 | 1281.62 | 0.00 | T1 1.5R @ 1276.24 |
| Target hit | 2026-04-29 14:15:00 | 1275.00 | 1274.57 | 0.00 | Trail-exit close>VWAP |

### Cycle 15 — SELL (started 2026-04-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:40:00 | 1271.50 | 1274.66 | 0.00 | ORB-short ORB[1274.60,1280.00] vol=5.7x ATR=2.70 |
| Stop hit — per-position SL triggered | 2026-04-30 10:45:00 | 1274.20 | 1274.65 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-05-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:55:00 | 1272.00 | 1282.45 | 0.00 | ORB-short ORB[1289.90,1304.00] vol=2.0x ATR=3.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:40:00 | 1266.50 | 1275.87 | 0.00 | T1 1.5R @ 1266.50 |
| Stop hit — per-position SL triggered | 2026-05-05 14:55:00 | 1272.00 | 1269.81 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-05-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:45:00 | 1261.40 | 1273.96 | 0.00 | ORB-short ORB[1275.90,1289.90] vol=2.4x ATR=3.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:55:00 | 1256.44 | 1270.98 | 0.00 | T1 1.5R @ 1256.44 |
| Target hit | 2026-05-06 15:20:00 | 1250.90 | 1256.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — SELL (started 2026-05-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 10:40:00 | 1243.60 | 1255.13 | 0.00 | ORB-short ORB[1252.90,1270.90] vol=2.7x ATR=4.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 11:10:00 | 1237.59 | 1252.56 | 0.00 | T1 1.5R @ 1237.59 |
| Stop hit — per-position SL triggered | 2026-05-07 11:25:00 | 1243.60 | 1252.14 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:25:00 | 1133.20 | 2026-02-09 11:05:00 | 1140.28 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-02-09 10:25:00 | 1133.20 | 2026-02-09 11:10:00 | 1133.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-10 09:40:00 | 1177.60 | 2026-02-10 10:05:00 | 1170.16 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest1 | 2026-02-18 10:50:00 | 1228.00 | 2026-02-18 10:55:00 | 1223.65 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-19 09:35:00 | 1208.00 | 2026-02-19 09:40:00 | 1202.51 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-02-19 09:35:00 | 1208.00 | 2026-02-19 10:25:00 | 1208.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-24 10:50:00 | 1212.90 | 2026-02-24 12:05:00 | 1216.88 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-03-16 10:50:00 | 1172.00 | 2026-03-16 14:20:00 | 1177.09 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-03-16 10:50:00 | 1172.00 | 2026-03-16 15:10:00 | 1172.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-18 09:30:00 | 1133.50 | 2026-03-18 10:35:00 | 1128.80 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-06 09:35:00 | 1290.00 | 2026-04-06 09:40:00 | 1284.84 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-17 11:00:00 | 1328.40 | 2026-04-17 11:10:00 | 1336.29 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2026-04-17 11:00:00 | 1328.40 | 2026-04-17 12:05:00 | 1328.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 11:10:00 | 1308.00 | 2026-04-21 11:20:00 | 1304.06 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-04-24 09:50:00 | 1273.90 | 2026-04-24 10:00:00 | 1278.02 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-04-27 10:10:00 | 1288.80 | 2026-04-27 10:15:00 | 1292.62 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-28 11:10:00 | 1295.00 | 2026-04-28 11:15:00 | 1291.27 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-04-29 11:00:00 | 1279.00 | 2026-04-29 12:45:00 | 1276.24 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2026-04-29 11:00:00 | 1279.00 | 2026-04-29 14:15:00 | 1275.00 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2026-04-30 10:40:00 | 1271.50 | 2026-04-30 10:45:00 | 1274.20 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-05-05 10:55:00 | 1272.00 | 2026-05-05 11:40:00 | 1266.50 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-05-05 10:55:00 | 1272.00 | 2026-05-05 14:55:00 | 1272.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-06 10:45:00 | 1261.40 | 2026-05-06 10:55:00 | 1256.44 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-05-06 10:45:00 | 1261.40 | 2026-05-06 15:20:00 | 1250.90 | TARGET_HIT | 0.50 | 0.83% |
| SELL | retest1 | 2026-05-07 10:40:00 | 1243.60 | 2026-05-07 11:10:00 | 1237.59 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-05-07 10:40:00 | 1243.60 | 2026-05-07 11:25:00 | 1243.60 | STOP_HIT | 0.50 | 0.00% |
