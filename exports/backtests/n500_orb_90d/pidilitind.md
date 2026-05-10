# Pidilite Industries Ltd. (PIDILITIND)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1472.00
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
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 18
- **Target hits / Stop hits / Partials:** 3 / 18 / 7
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 2.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 6 | 50.0% | 2 | 6 | 4 | 0.38% | 4.6% |
| BUY @ 2nd Alert (retest1) | 12 | 6 | 50.0% | 2 | 6 | 4 | 0.38% | 4.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 16 | 4 | 25.0% | 1 | 12 | 3 | -0.10% | -1.6% |
| SELL @ 2nd Alert (retest1) | 16 | 4 | 25.0% | 1 | 12 | 3 | -0.10% | -1.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 28 | 10 | 35.7% | 3 | 18 | 7 | 0.11% | 3.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:55:00 | 1477.90 | 1482.30 | 0.00 | ORB-short ORB[1481.60,1494.00] vol=2.4x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 11:10:00 | 1474.25 | 1481.61 | 0.00 | T1 1.5R @ 1474.25 |
| Target hit | 2026-02-10 14:25:00 | 1475.90 | 1475.65 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — BUY (started 2026-02-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 11:00:00 | 1494.60 | 1486.91 | 0.00 | ORB-long ORB[1475.00,1491.60] vol=1.9x ATR=3.32 |
| Stop hit — per-position SL triggered | 2026-02-11 11:05:00 | 1491.28 | 1487.16 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:50:00 | 1474.30 | 1482.56 | 0.00 | ORB-short ORB[1480.30,1489.60] vol=1.7x ATR=3.32 |
| Stop hit — per-position SL triggered | 2026-02-13 11:25:00 | 1477.62 | 1481.50 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:45:00 | 1491.40 | 1486.83 | 0.00 | ORB-long ORB[1477.40,1490.00] vol=3.6x ATR=3.17 |
| Stop hit — per-position SL triggered | 2026-02-17 11:10:00 | 1488.23 | 1487.89 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 11:10:00 | 1480.60 | 1477.45 | 0.00 | ORB-long ORB[1470.00,1479.30] vol=1.6x ATR=2.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:40:00 | 1484.50 | 1479.03 | 0.00 | T1 1.5R @ 1484.50 |
| Stop hit — per-position SL triggered | 2026-02-24 12:40:00 | 1480.60 | 1480.16 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 11:00:00 | 1485.00 | 1487.02 | 0.00 | ORB-short ORB[1488.10,1495.50] vol=4.6x ATR=2.34 |
| Stop hit — per-position SL triggered | 2026-02-25 11:05:00 | 1487.34 | 1487.00 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:45:00 | 1498.90 | 1497.09 | 0.00 | ORB-long ORB[1487.10,1497.80] vol=8.0x ATR=3.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:05:00 | 1503.42 | 1499.01 | 0.00 | T1 1.5R @ 1503.42 |
| Stop hit — per-position SL triggered | 2026-02-26 11:40:00 | 1498.90 | 1501.08 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-02-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:40:00 | 1505.90 | 1506.76 | 0.00 | ORB-short ORB[1507.10,1518.90] vol=1.7x ATR=3.08 |
| Stop hit — per-position SL triggered | 2026-02-27 11:05:00 | 1508.98 | 1506.69 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:50:00 | 1426.20 | 1432.15 | 0.00 | ORB-short ORB[1432.60,1446.90] vol=3.4x ATR=3.02 |
| Stop hit — per-position SL triggered | 2026-03-05 11:15:00 | 1429.22 | 1431.52 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 11:10:00 | 1425.00 | 1430.49 | 0.00 | ORB-short ORB[1427.70,1438.60] vol=1.8x ATR=2.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:15:00 | 1420.88 | 1430.02 | 0.00 | T1 1.5R @ 1420.88 |
| Stop hit — per-position SL triggered | 2026-03-11 11:20:00 | 1425.00 | 1428.97 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 10:00:00 | 1362.60 | 1360.90 | 0.00 | ORB-long ORB[1345.60,1361.30] vol=1.7x ATR=4.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 10:40:00 | 1369.20 | 1361.78 | 0.00 | T1 1.5R @ 1369.20 |
| Target hit | 2026-03-18 15:20:00 | 1382.80 | 1377.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2026-03-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-30 11:10:00 | 1282.00 | 1297.17 | 0.00 | ORB-short ORB[1296.10,1311.00] vol=1.7x ATR=3.99 |
| Stop hit — per-position SL triggered | 2026-03-30 12:00:00 | 1285.99 | 1292.62 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 11:00:00 | 1345.20 | 1350.00 | 0.00 | ORB-short ORB[1346.90,1365.00] vol=9.0x ATR=3.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 11:25:00 | 1339.76 | 1349.15 | 0.00 | T1 1.5R @ 1339.76 |
| Stop hit — per-position SL triggered | 2026-04-10 12:00:00 | 1345.20 | 1347.18 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:50:00 | 1341.30 | 1334.95 | 0.00 | ORB-long ORB[1314.30,1334.20] vol=5.5x ATR=5.13 |
| Stop hit — per-position SL triggered | 2026-04-13 11:05:00 | 1336.17 | 1336.27 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:55:00 | 1355.80 | 1343.21 | 0.00 | ORB-long ORB[1326.00,1343.90] vol=2.0x ATR=3.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:00:00 | 1361.61 | 1345.76 | 0.00 | T1 1.5R @ 1361.61 |
| Target hit | 2026-04-17 15:20:00 | 1393.10 | 1379.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2026-04-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:40:00 | 1409.20 | 1395.12 | 0.00 | ORB-long ORB[1380.00,1394.60] vol=2.2x ATR=4.19 |
| Stop hit — per-position SL triggered | 2026-04-21 11:30:00 | 1405.01 | 1404.58 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-04-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:50:00 | 1393.60 | 1398.36 | 0.00 | ORB-short ORB[1395.00,1408.90] vol=2.4x ATR=3.62 |
| Stop hit — per-position SL triggered | 2026-04-24 10:45:00 | 1397.22 | 1397.98 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-04-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 11:05:00 | 1359.40 | 1368.30 | 0.00 | ORB-short ORB[1368.80,1381.10] vol=1.7x ATR=3.26 |
| Stop hit — per-position SL triggered | 2026-04-30 11:40:00 | 1362.66 | 1365.76 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2026-05-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 10:30:00 | 1373.80 | 1377.33 | 0.00 | ORB-short ORB[1376.70,1388.60] vol=2.8x ATR=4.23 |
| Stop hit — per-position SL triggered | 2026-05-04 10:35:00 | 1378.03 | 1377.54 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2026-05-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:00:00 | 1350.50 | 1357.84 | 0.00 | ORB-short ORB[1357.80,1369.00] vol=2.9x ATR=2.99 |
| Stop hit — per-position SL triggered | 2026-05-05 11:15:00 | 1353.49 | 1355.76 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2026-05-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:35:00 | 1476.90 | 1497.84 | 0.00 | ORB-short ORB[1492.70,1515.00] vol=2.4x ATR=8.29 |
| Stop hit — per-position SL triggered | 2026-05-08 10:15:00 | 1485.19 | 1493.15 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 10:55:00 | 1477.90 | 2026-02-10 11:10:00 | 1474.25 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2026-02-10 10:55:00 | 1477.90 | 2026-02-10 14:25:00 | 1475.90 | TARGET_HIT | 0.50 | 0.14% |
| BUY | retest1 | 2026-02-11 11:00:00 | 1494.60 | 2026-02-11 11:05:00 | 1491.28 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-13 10:50:00 | 1474.30 | 2026-02-13 11:25:00 | 1477.62 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-02-17 10:45:00 | 1491.40 | 2026-02-17 11:10:00 | 1488.23 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-02-24 11:10:00 | 1480.60 | 2026-02-24 11:40:00 | 1484.50 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2026-02-24 11:10:00 | 1480.60 | 2026-02-24 12:40:00 | 1480.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-25 11:00:00 | 1485.00 | 2026-02-25 11:05:00 | 1487.34 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2026-02-26 10:45:00 | 1498.90 | 2026-02-26 11:05:00 | 1503.42 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-02-26 10:45:00 | 1498.90 | 2026-02-26 11:40:00 | 1498.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 10:40:00 | 1505.90 | 2026-02-27 11:05:00 | 1508.98 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-03-05 10:50:00 | 1426.20 | 2026-03-05 11:15:00 | 1429.22 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-03-11 11:10:00 | 1425.00 | 2026-03-11 11:15:00 | 1420.88 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2026-03-11 11:10:00 | 1425.00 | 2026-03-11 11:20:00 | 1425.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-18 10:00:00 | 1362.60 | 2026-03-18 10:40:00 | 1369.20 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-03-18 10:00:00 | 1362.60 | 2026-03-18 15:20:00 | 1382.80 | TARGET_HIT | 0.50 | 1.48% |
| SELL | retest1 | 2026-03-30 11:10:00 | 1282.00 | 2026-03-30 12:00:00 | 1285.99 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-04-10 11:00:00 | 1345.20 | 2026-04-10 11:25:00 | 1339.76 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-04-10 11:00:00 | 1345.20 | 2026-04-10 12:00:00 | 1345.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-13 10:50:00 | 1341.30 | 2026-04-13 11:05:00 | 1336.17 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-17 09:55:00 | 1355.80 | 2026-04-17 10:00:00 | 1361.61 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-04-17 09:55:00 | 1355.80 | 2026-04-17 15:20:00 | 1393.10 | TARGET_HIT | 0.50 | 2.75% |
| BUY | retest1 | 2026-04-21 09:40:00 | 1409.20 | 2026-04-21 11:30:00 | 1405.01 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-04-24 09:50:00 | 1393.60 | 2026-04-24 10:45:00 | 1397.22 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-30 11:05:00 | 1359.40 | 2026-04-30 11:40:00 | 1362.66 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-05-04 10:30:00 | 1373.80 | 2026-05-04 10:35:00 | 1378.03 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-05-05 11:00:00 | 1350.50 | 2026-05-05 11:15:00 | 1353.49 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-05-08 09:35:00 | 1476.90 | 2026-05-08 10:15:00 | 1485.19 | STOP_HIT | 1.00 | -0.56% |
