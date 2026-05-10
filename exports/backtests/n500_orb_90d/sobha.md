# Sobha Ltd. (SOBHA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1425.00
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
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 18
- **Target hits / Stop hits / Partials:** 0 / 18 / 2
- **Avg / median % per leg:** -0.23% / -0.33%
- **Sum % (uncompounded):** -4.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 1 | 12.5% | 0 | 7 | 1 | -0.18% | -1.5% |
| BUY @ 2nd Alert (retest1) | 8 | 1 | 12.5% | 0 | 7 | 1 | -0.18% | -1.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 1 | 8.3% | 0 | 11 | 1 | -0.27% | -3.2% |
| SELL @ 2nd Alert (retest1) | 12 | 1 | 8.3% | 0 | 11 | 1 | -0.27% | -3.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 20 | 2 | 10.0% | 0 | 18 | 2 | -0.23% | -4.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 09:50:00 | 1482.50 | 1491.97 | 0.00 | ORB-short ORB[1488.10,1509.70] vol=1.7x ATR=7.09 |
| Stop hit — per-position SL triggered | 2026-02-16 10:00:00 | 1489.59 | 1491.80 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-18 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:20:00 | 1475.00 | 1481.53 | 0.00 | ORB-short ORB[1475.10,1486.90] vol=2.3x ATR=4.17 |
| Stop hit — per-position SL triggered | 2026-02-18 11:00:00 | 1479.17 | 1478.80 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:35:00 | 1482.60 | 1487.85 | 0.00 | ORB-short ORB[1484.60,1499.80] vol=1.8x ATR=4.01 |
| Stop hit — per-position SL triggered | 2026-02-19 10:40:00 | 1486.61 | 1487.82 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 11:00:00 | 1526.90 | 1518.70 | 0.00 | ORB-long ORB[1508.40,1521.00] vol=3.0x ATR=3.54 |
| Stop hit — per-position SL triggered | 2026-02-23 11:20:00 | 1523.36 | 1519.90 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 1498.20 | 1504.10 | 0.00 | ORB-short ORB[1501.00,1513.10] vol=1.6x ATR=4.37 |
| Stop hit — per-position SL triggered | 2026-02-24 09:35:00 | 1502.57 | 1501.49 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:40:00 | 1465.40 | 1458.06 | 0.00 | ORB-long ORB[1443.10,1455.80] vol=1.8x ATR=6.29 |
| Stop hit — per-position SL triggered | 2026-02-26 11:30:00 | 1459.11 | 1461.87 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:45:00 | 1338.90 | 1311.40 | 0.00 | ORB-long ORB[1295.10,1310.00] vol=5.0x ATR=5.81 |
| Stop hit — per-position SL triggered | 2026-03-10 10:50:00 | 1333.09 | 1316.28 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 11:00:00 | 1342.20 | 1336.73 | 0.00 | ORB-long ORB[1321.00,1341.00] vol=3.9x ATR=3.46 |
| Stop hit — per-position SL triggered | 2026-03-11 13:15:00 | 1338.74 | 1338.48 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-24 10:45:00 | 1224.30 | 1211.76 | 0.00 | ORB-long ORB[1207.10,1221.10] vol=2.4x ATR=5.65 |
| Stop hit — per-position SL triggered | 2026-03-24 10:50:00 | 1218.65 | 1211.56 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:55:00 | 1276.70 | 1279.39 | 0.00 | ORB-short ORB[1282.50,1294.30] vol=3.1x ATR=6.28 |
| Stop hit — per-position SL triggered | 2026-04-09 10:15:00 | 1282.98 | 1279.27 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:40:00 | 1317.50 | 1327.98 | 0.00 | ORB-short ORB[1321.00,1339.90] vol=3.9x ATR=4.38 |
| Stop hit — per-position SL triggered | 2026-04-16 11:25:00 | 1321.88 | 1321.86 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:05:00 | 1333.80 | 1326.14 | 0.00 | ORB-long ORB[1313.40,1332.90] vol=2.0x ATR=5.06 |
| Stop hit — per-position SL triggered | 2026-04-17 10:10:00 | 1328.74 | 1326.79 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-20 09:35:00 | 1315.90 | 1325.06 | 0.00 | ORB-short ORB[1317.90,1337.40] vol=2.1x ATR=4.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-20 09:45:00 | 1308.59 | 1323.21 | 0.00 | T1 1.5R @ 1308.59 |
| Stop hit — per-position SL triggered | 2026-04-20 09:50:00 | 1315.90 | 1322.45 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:55:00 | 1355.80 | 1367.20 | 0.00 | ORB-short ORB[1365.70,1382.10] vol=1.8x ATR=5.70 |
| Stop hit — per-position SL triggered | 2026-04-22 11:05:00 | 1361.50 | 1366.16 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 10:50:00 | 1410.30 | 1415.75 | 0.00 | ORB-short ORB[1411.70,1430.30] vol=2.9x ATR=5.41 |
| Stop hit — per-position SL triggered | 2026-04-27 11:30:00 | 1415.71 | 1414.38 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-04-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:45:00 | 1416.50 | 1421.13 | 0.00 | ORB-short ORB[1419.50,1435.00] vol=2.8x ATR=4.80 |
| Stop hit — per-position SL triggered | 2026-04-28 10:20:00 | 1421.30 | 1419.91 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-04-29 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:30:00 | 1439.00 | 1452.63 | 0.00 | ORB-short ORB[1442.50,1464.10] vol=1.6x ATR=6.74 |
| Stop hit — per-position SL triggered | 2026-04-29 11:25:00 | 1445.74 | 1451.04 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-05-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:35:00 | 1462.80 | 1455.91 | 0.00 | ORB-long ORB[1439.90,1460.00] vol=2.3x ATR=7.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 09:55:00 | 1473.49 | 1462.85 | 0.00 | T1 1.5R @ 1473.49 |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 1462.80 | 1463.85 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-16 09:50:00 | 1482.50 | 2026-02-16 10:00:00 | 1489.59 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2026-02-18 10:20:00 | 1475.00 | 2026-02-18 11:00:00 | 1479.17 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-19 10:35:00 | 1482.60 | 2026-02-19 10:40:00 | 1486.61 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-23 11:00:00 | 1526.90 | 2026-02-23 11:20:00 | 1523.36 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-02-24 09:30:00 | 1498.20 | 2026-02-24 09:35:00 | 1502.57 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-26 09:40:00 | 1465.40 | 2026-02-26 11:30:00 | 1459.11 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-03-10 10:45:00 | 1338.90 | 2026-03-10 10:50:00 | 1333.09 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-03-11 11:00:00 | 1342.20 | 2026-03-11 13:15:00 | 1338.74 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-03-24 10:45:00 | 1224.30 | 2026-03-24 10:50:00 | 1218.65 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-04-09 09:55:00 | 1276.70 | 2026-04-09 10:15:00 | 1282.98 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-04-16 09:40:00 | 1317.50 | 2026-04-16 11:25:00 | 1321.88 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-17 10:05:00 | 1333.80 | 2026-04-17 10:10:00 | 1328.74 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-04-20 09:35:00 | 1315.90 | 2026-04-20 09:45:00 | 1308.59 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-04-20 09:35:00 | 1315.90 | 2026-04-20 09:50:00 | 1315.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-22 10:55:00 | 1355.80 | 2026-04-22 11:05:00 | 1361.50 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-04-27 10:50:00 | 1410.30 | 2026-04-27 11:30:00 | 1415.71 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-04-28 09:45:00 | 1416.50 | 2026-04-28 10:20:00 | 1421.30 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-04-29 10:30:00 | 1439.00 | 2026-04-29 11:25:00 | 1445.74 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-05-04 09:35:00 | 1462.80 | 2026-05-04 09:55:00 | 1473.49 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2026-05-04 09:35:00 | 1462.80 | 2026-05-04 10:15:00 | 1462.80 | STOP_HIT | 0.50 | 0.00% |
