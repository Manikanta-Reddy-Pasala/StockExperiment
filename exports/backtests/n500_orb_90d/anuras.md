# Anupam Rasayan India Ltd. (ANURAS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1369.00
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
| ENTRY1 | 21 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 20
- **Target hits / Stop hits / Partials:** 1 / 20 / 5
- **Avg / median % per leg:** -0.11% / -0.26%
- **Sum % (uncompounded):** -2.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 4 | 30.8% | 1 | 9 | 3 | -0.04% | -0.6% |
| BUY @ 2nd Alert (retest1) | 13 | 4 | 30.8% | 1 | 9 | 3 | -0.04% | -0.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 13 | 2 | 15.4% | 0 | 11 | 2 | -0.17% | -2.2% |
| SELL @ 2nd Alert (retest1) | 13 | 2 | 15.4% | 0 | 11 | 2 | -0.17% | -2.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 26 | 6 | 23.1% | 1 | 20 | 5 | -0.11% | -2.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:35:00 | 1338.80 | 1327.75 | 0.00 | ORB-long ORB[1315.30,1326.20] vol=2.9x ATR=6.58 |
| Stop hit — per-position SL triggered | 2026-02-09 10:45:00 | 1332.22 | 1330.27 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 1337.30 | 1330.13 | 0.00 | ORB-long ORB[1315.70,1333.90] vol=3.5x ATR=4.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:40:00 | 1344.39 | 1333.11 | 0.00 | T1 1.5R @ 1344.39 |
| Stop hit — per-position SL triggered | 2026-02-10 09:55:00 | 1337.30 | 1334.01 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:20:00 | 1333.40 | 1323.97 | 0.00 | ORB-long ORB[1311.10,1329.00] vol=1.7x ATR=4.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 10:30:00 | 1339.82 | 1325.77 | 0.00 | T1 1.5R @ 1339.82 |
| Stop hit — per-position SL triggered | 2026-02-11 10:40:00 | 1333.40 | 1328.39 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:10:00 | 1367.60 | 1356.03 | 0.00 | ORB-long ORB[1344.10,1362.80] vol=2.1x ATR=4.89 |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 1362.71 | 1356.78 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:15:00 | 1340.70 | 1351.57 | 0.00 | ORB-short ORB[1348.70,1368.20] vol=2.1x ATR=6.31 |
| Stop hit — per-position SL triggered | 2026-02-13 11:10:00 | 1347.01 | 1349.71 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:55:00 | 1282.90 | 1276.88 | 0.00 | ORB-long ORB[1262.00,1279.00] vol=2.1x ATR=3.30 |
| Stop hit — per-position SL triggered | 2026-02-18 11:05:00 | 1279.60 | 1277.01 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-20 10:35:00 | 1305.70 | 1315.73 | 0.00 | ORB-short ORB[1311.50,1329.10] vol=1.9x ATR=5.27 |
| Stop hit — per-position SL triggered | 2026-02-20 10:50:00 | 1310.97 | 1315.02 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-02-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:10:00 | 1257.00 | 1258.53 | 0.00 | ORB-short ORB[1261.90,1269.50] vol=4.6x ATR=3.24 |
| Stop hit — per-position SL triggered | 2026-02-26 10:40:00 | 1260.24 | 1259.41 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-02-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:20:00 | 1239.70 | 1245.54 | 0.00 | ORB-short ORB[1246.00,1253.60] vol=1.8x ATR=2.96 |
| Stop hit — per-position SL triggered | 2026-02-27 10:50:00 | 1242.66 | 1245.07 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 1225.50 | 1235.47 | 0.00 | ORB-short ORB[1237.30,1251.50] vol=2.1x ATR=3.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:55:00 | 1220.41 | 1232.77 | 0.00 | T1 1.5R @ 1220.41 |
| Stop hit — per-position SL triggered | 2026-03-06 11:15:00 | 1225.50 | 1229.84 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:50:00 | 1214.00 | 1227.44 | 0.00 | ORB-short ORB[1219.70,1230.10] vol=2.3x ATR=5.04 |
| Stop hit — per-position SL triggered | 2026-03-16 11:30:00 | 1219.04 | 1222.61 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:55:00 | 1247.80 | 1241.82 | 0.00 | ORB-long ORB[1227.90,1246.20] vol=3.1x ATR=3.55 |
| Stop hit — per-position SL triggered | 2026-03-17 11:20:00 | 1244.25 | 1242.30 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-03-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 10:50:00 | 1223.10 | 1225.15 | 0.00 | ORB-short ORB[1227.10,1244.40] vol=1.6x ATR=3.66 |
| Stop hit — per-position SL triggered | 2026-03-23 11:05:00 | 1226.76 | 1225.21 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-07 09:40:00 | 1229.00 | 1233.49 | 0.00 | ORB-short ORB[1230.00,1243.00] vol=1.7x ATR=4.99 |
| Stop hit — per-position SL triggered | 2026-04-07 09:55:00 | 1233.99 | 1233.07 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:40:00 | 1297.10 | 1293.34 | 0.00 | ORB-long ORB[1281.40,1296.00] vol=2.2x ATR=4.10 |
| Stop hit — per-position SL triggered | 2026-04-15 09:50:00 | 1293.00 | 1293.36 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-04-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:25:00 | 1277.40 | 1279.17 | 0.00 | ORB-short ORB[1285.10,1294.90] vol=2.7x ATR=3.35 |
| Stop hit — per-position SL triggered | 2026-04-16 10:30:00 | 1280.75 | 1279.18 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:40:00 | 1330.70 | 1321.53 | 0.00 | ORB-long ORB[1311.10,1321.90] vol=5.2x ATR=4.44 |
| Stop hit — per-position SL triggered | 2026-04-21 09:45:00 | 1326.26 | 1323.10 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-04-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 11:05:00 | 1343.10 | 1336.13 | 0.00 | ORB-long ORB[1315.10,1329.90] vol=3.7x ATR=4.44 |
| Target hit | 2026-04-22 15:20:00 | 1344.00 | 1341.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — BUY (started 2026-04-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:05:00 | 1362.30 | 1357.20 | 0.00 | ORB-long ORB[1345.60,1359.00] vol=2.9x ATR=3.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 10:20:00 | 1367.60 | 1359.68 | 0.00 | T1 1.5R @ 1367.60 |
| Stop hit — per-position SL triggered | 2026-04-23 10:50:00 | 1362.30 | 1360.40 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2026-04-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 10:10:00 | 1343.70 | 1345.86 | 0.00 | ORB-short ORB[1344.90,1364.40] vol=3.1x ATR=4.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 11:05:00 | 1336.79 | 1344.89 | 0.00 | T1 1.5R @ 1336.79 |
| Stop hit — per-position SL triggered | 2026-04-27 13:00:00 | 1343.70 | 1338.60 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2026-04-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 09:50:00 | 1304.00 | 1309.56 | 0.00 | ORB-short ORB[1306.20,1320.00] vol=3.1x ATR=4.84 |
| Stop hit — per-position SL triggered | 2026-04-29 12:10:00 | 1308.84 | 1308.25 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:35:00 | 1338.80 | 2026-02-09 10:45:00 | 1332.22 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2026-02-10 09:30:00 | 1337.30 | 2026-02-10 09:40:00 | 1344.39 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-02-10 09:30:00 | 1337.30 | 2026-02-10 09:55:00 | 1337.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-11 10:20:00 | 1333.40 | 2026-02-11 10:30:00 | 1339.82 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-02-11 10:20:00 | 1333.40 | 2026-02-11 10:40:00 | 1333.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-12 10:10:00 | 1367.60 | 2026-02-12 10:15:00 | 1362.71 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-13 10:15:00 | 1340.70 | 2026-02-13 11:10:00 | 1347.01 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-02-18 10:55:00 | 1282.90 | 2026-02-18 11:05:00 | 1279.60 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-20 10:35:00 | 1305.70 | 2026-02-20 10:50:00 | 1310.97 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-02-26 10:10:00 | 1257.00 | 2026-02-26 10:40:00 | 1260.24 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-27 10:20:00 | 1239.70 | 2026-02-27 10:50:00 | 1242.66 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-03-06 10:45:00 | 1225.50 | 2026-03-06 10:55:00 | 1220.41 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-03-06 10:45:00 | 1225.50 | 2026-03-06 11:15:00 | 1225.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-16 10:50:00 | 1214.00 | 2026-03-16 11:30:00 | 1219.04 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-03-17 10:55:00 | 1247.80 | 2026-03-17 11:20:00 | 1244.25 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-03-23 10:50:00 | 1223.10 | 2026-03-23 11:05:00 | 1226.76 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-04-07 09:40:00 | 1229.00 | 2026-04-07 09:55:00 | 1233.99 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-15 09:40:00 | 1297.10 | 2026-04-15 09:50:00 | 1293.00 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-04-16 10:25:00 | 1277.40 | 2026-04-16 10:30:00 | 1280.75 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-04-21 09:40:00 | 1330.70 | 2026-04-21 09:45:00 | 1326.26 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-22 11:05:00 | 1343.10 | 2026-04-22 15:20:00 | 1344.00 | TARGET_HIT | 1.00 | 0.07% |
| BUY | retest1 | 2026-04-23 10:05:00 | 1362.30 | 2026-04-23 10:20:00 | 1367.60 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-04-23 10:05:00 | 1362.30 | 2026-04-23 10:50:00 | 1362.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-27 10:10:00 | 1343.70 | 2026-04-27 11:05:00 | 1336.79 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-04-27 10:10:00 | 1343.70 | 2026-04-27 13:00:00 | 1343.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-29 09:50:00 | 1304.00 | 2026-04-29 12:10:00 | 1308.84 | STOP_HIT | 1.00 | -0.37% |
