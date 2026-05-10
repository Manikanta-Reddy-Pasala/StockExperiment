# Wockhardt Ltd. (WOCKPHARMA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1611.50
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
| ENTRY1 | 15 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 12
- **Target hits / Stop hits / Partials:** 3 / 12 / 7
- **Avg / median % per leg:** 0.20% / 0.00%
- **Sum % (uncompounded):** 4.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 2 | 16.7% | 0 | 10 | 2 | -0.15% | -1.8% |
| BUY @ 2nd Alert (retest1) | 12 | 2 | 16.7% | 0 | 10 | 2 | -0.15% | -1.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 8 | 80.0% | 3 | 2 | 5 | 0.61% | 6.1% |
| SELL @ 2nd Alert (retest1) | 10 | 8 | 80.0% | 3 | 2 | 5 | 0.61% | 6.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 10 | 45.5% | 3 | 12 | 7 | 0.20% | 4.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 11:10:00 | 1393.00 | 1374.90 | 0.00 | ORB-long ORB[1363.00,1372.90] vol=2.1x ATR=4.68 |
| Stop hit — per-position SL triggered | 2026-02-13 11:15:00 | 1388.32 | 1375.25 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:05:00 | 1402.00 | 1396.72 | 0.00 | ORB-long ORB[1381.00,1400.00] vol=1.9x ATR=3.55 |
| Stop hit — per-position SL triggered | 2026-02-17 10:15:00 | 1398.45 | 1397.00 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:40:00 | 1415.60 | 1426.33 | 0.00 | ORB-short ORB[1422.10,1436.90] vol=2.1x ATR=4.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:20:00 | 1408.54 | 1424.83 | 0.00 | T1 1.5R @ 1408.54 |
| Target hit | 2026-02-19 15:20:00 | 1389.10 | 1410.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-02-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:35:00 | 1371.50 | 1378.80 | 0.00 | ORB-short ORB[1374.00,1390.50] vol=2.2x ATR=3.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 10:45:00 | 1366.12 | 1376.74 | 0.00 | T1 1.5R @ 1366.12 |
| Target hit | 2026-02-23 15:20:00 | 1365.10 | 1366.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2026-02-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:45:00 | 1371.00 | 1365.82 | 0.00 | ORB-long ORB[1351.20,1370.30] vol=2.5x ATR=3.93 |
| Stop hit — per-position SL triggered | 2026-02-25 10:50:00 | 1367.07 | 1365.85 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:50:00 | 1401.60 | 1389.30 | 0.00 | ORB-long ORB[1369.50,1389.80] vol=2.8x ATR=5.90 |
| Stop hit — per-position SL triggered | 2026-02-26 09:55:00 | 1395.70 | 1390.07 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 10:05:00 | 1313.00 | 1301.19 | 0.00 | ORB-long ORB[1288.90,1306.60] vol=2.3x ATR=6.95 |
| Stop hit — per-position SL triggered | 2026-03-05 10:20:00 | 1306.05 | 1303.36 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:35:00 | 1215.30 | 1205.29 | 0.00 | ORB-long ORB[1192.00,1210.00] vol=1.6x ATR=5.18 |
| Stop hit — per-position SL triggered | 2026-03-18 09:40:00 | 1210.12 | 1207.15 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:30:00 | 1214.10 | 1205.90 | 0.00 | ORB-long ORB[1193.20,1211.00] vol=2.1x ATR=5.01 |
| Stop hit — per-position SL triggered | 2026-03-20 09:35:00 | 1209.09 | 1206.70 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 10:45:00 | 1365.00 | 1372.27 | 0.00 | ORB-short ORB[1366.10,1382.40] vol=3.2x ATR=5.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 13:15:00 | 1357.40 | 1368.70 | 0.00 | T1 1.5R @ 1357.40 |
| Target hit | 2026-04-10 15:20:00 | 1344.90 | 1354.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2026-04-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 11:10:00 | 1330.60 | 1320.53 | 0.00 | ORB-long ORB[1307.50,1326.20] vol=1.8x ATR=4.50 |
| Stop hit — per-position SL triggered | 2026-04-13 13:00:00 | 1326.10 | 1322.58 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 11:05:00 | 1381.00 | 1370.97 | 0.00 | ORB-long ORB[1360.00,1372.00] vol=2.6x ATR=4.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 11:10:00 | 1387.97 | 1373.63 | 0.00 | T1 1.5R @ 1387.97 |
| Stop hit — per-position SL triggered | 2026-04-17 11:15:00 | 1381.00 | 1374.19 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:55:00 | 1428.60 | 1418.26 | 0.00 | ORB-long ORB[1405.00,1424.00] vol=1.7x ATR=6.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 10:00:00 | 1438.40 | 1420.34 | 0.00 | T1 1.5R @ 1438.40 |
| Stop hit — per-position SL triggered | 2026-04-22 10:55:00 | 1428.60 | 1424.36 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:30:00 | 1419.80 | 1425.36 | 0.00 | ORB-short ORB[1423.80,1437.40] vol=1.6x ATR=4.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:10:00 | 1413.13 | 1420.60 | 0.00 | T1 1.5R @ 1413.13 |
| Stop hit — per-position SL triggered | 2026-04-28 10:20:00 | 1419.80 | 1420.32 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:35:00 | 1411.00 | 1420.29 | 0.00 | ORB-short ORB[1417.90,1437.00] vol=1.9x ATR=3.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:45:00 | 1405.08 | 1418.55 | 0.00 | T1 1.5R @ 1405.08 |
| Stop hit — per-position SL triggered | 2026-04-29 10:50:00 | 1411.00 | 1418.51 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-13 11:10:00 | 1393.00 | 2026-02-13 11:15:00 | 1388.32 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-17 10:05:00 | 1402.00 | 2026-02-17 10:15:00 | 1398.45 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-02-19 10:40:00 | 1415.60 | 2026-02-19 11:20:00 | 1408.54 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-02-19 10:40:00 | 1415.60 | 2026-02-19 15:20:00 | 1389.10 | TARGET_HIT | 0.50 | 1.87% |
| SELL | retest1 | 2026-02-23 10:35:00 | 1371.50 | 2026-02-23 10:45:00 | 1366.12 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-02-23 10:35:00 | 1371.50 | 2026-02-23 15:20:00 | 1365.10 | TARGET_HIT | 0.50 | 0.47% |
| BUY | retest1 | 2026-02-25 10:45:00 | 1371.00 | 2026-02-25 10:50:00 | 1367.07 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-26 09:50:00 | 1401.60 | 2026-02-26 09:55:00 | 1395.70 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-03-05 10:05:00 | 1313.00 | 2026-03-05 10:20:00 | 1306.05 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2026-03-18 09:35:00 | 1215.30 | 2026-03-18 09:40:00 | 1210.12 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-03-20 09:30:00 | 1214.10 | 2026-03-20 09:35:00 | 1209.09 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-04-10 10:45:00 | 1365.00 | 2026-04-10 13:15:00 | 1357.40 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-04-10 10:45:00 | 1365.00 | 2026-04-10 15:20:00 | 1344.90 | TARGET_HIT | 0.50 | 1.47% |
| BUY | retest1 | 2026-04-13 11:10:00 | 1330.60 | 2026-04-13 13:00:00 | 1326.10 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-17 11:05:00 | 1381.00 | 2026-04-17 11:10:00 | 1387.97 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-04-17 11:05:00 | 1381.00 | 2026-04-17 11:15:00 | 1381.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 09:55:00 | 1428.60 | 2026-04-22 10:00:00 | 1438.40 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2026-04-22 09:55:00 | 1428.60 | 2026-04-22 10:55:00 | 1428.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-28 09:30:00 | 1419.80 | 2026-04-28 10:10:00 | 1413.13 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-04-28 09:30:00 | 1419.80 | 2026-04-28 10:20:00 | 1419.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-29 10:35:00 | 1411.00 | 2026-04-29 10:45:00 | 1405.08 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-04-29 10:35:00 | 1411.00 | 2026-04-29 10:50:00 | 1411.00 | STOP_HIT | 0.50 | 0.00% |
