# Godrej Consumer Products Ltd. (GODREJCP)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1041.90
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
| ENTRY1 | 90 |
| ENTRY2 | 0 |
| PARTIAL | 38 |
| TARGET_HIT | 17 |
| STOP_HIT | 73 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 128 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 55 / 73
- **Target hits / Stop hits / Partials:** 17 / 73 / 38
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 19.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 54 | 23 | 42.6% | 8 | 31 | 15 | 0.17% | 9.4% |
| BUY @ 2nd Alert (retest1) | 54 | 23 | 42.6% | 8 | 31 | 15 | 0.17% | 9.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 74 | 32 | 43.2% | 9 | 42 | 23 | 0.13% | 9.7% |
| SELL @ 2nd Alert (retest1) | 74 | 32 | 43.2% | 9 | 42 | 23 | 0.13% | 9.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 128 | 55 | 43.0% | 17 | 73 | 38 | 0.15% | 19.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-13 11:00:00 | 1334.90 | 1323.48 | 0.00 | ORB-long ORB[1317.65,1327.60] vol=2.3x ATR=5.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-13 11:50:00 | 1342.91 | 1328.93 | 0.00 | T1 1.5R @ 1342.91 |
| Stop hit — per-position SL triggered | 2024-05-13 15:00:00 | 1334.90 | 1336.28 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-22 09:35:00 | 1304.30 | 1295.91 | 0.00 | ORB-long ORB[1283.70,1295.00] vol=1.8x ATR=3.88 |
| Stop hit — per-position SL triggered | 2024-05-22 09:40:00 | 1300.42 | 1297.16 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-31 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 10:20:00 | 1270.45 | 1273.77 | 0.00 | ORB-short ORB[1271.00,1289.80] vol=4.0x ATR=4.33 |
| Stop hit — per-position SL triggered | 2024-05-31 10:25:00 | 1274.78 | 1273.75 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-06-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 10:10:00 | 1420.30 | 1408.94 | 0.00 | ORB-long ORB[1391.55,1408.10] vol=1.6x ATR=5.55 |
| Stop hit — per-position SL triggered | 2024-06-07 10:15:00 | 1414.75 | 1409.71 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-06-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 09:40:00 | 1426.15 | 1431.26 | 0.00 | ORB-short ORB[1428.10,1447.80] vol=2.1x ATR=6.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 09:55:00 | 1415.79 | 1428.52 | 0.00 | T1 1.5R @ 1415.79 |
| Stop hit — per-position SL triggered | 2024-06-10 11:15:00 | 1426.15 | 1426.11 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 10:05:00 | 1399.00 | 1389.49 | 0.00 | ORB-long ORB[1387.90,1397.60] vol=1.5x ATR=4.59 |
| Stop hit — per-position SL triggered | 2024-06-18 10:10:00 | 1394.41 | 1389.69 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 10:45:00 | 1389.70 | 1394.67 | 0.00 | ORB-short ORB[1397.15,1410.75] vol=3.5x ATR=4.15 |
| Stop hit — per-position SL triggered | 2024-06-19 10:50:00 | 1393.85 | 1394.00 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:05:00 | 1388.20 | 1380.50 | 0.00 | ORB-long ORB[1370.25,1383.35] vol=2.2x ATR=3.83 |
| Stop hit — per-position SL triggered | 2024-06-26 10:10:00 | 1384.37 | 1380.75 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-07-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 10:30:00 | 1389.90 | 1381.25 | 0.00 | ORB-long ORB[1370.25,1382.70] vol=1.7x ATR=3.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 10:40:00 | 1395.49 | 1382.57 | 0.00 | T1 1.5R @ 1395.49 |
| Stop hit — per-position SL triggered | 2024-07-01 12:30:00 | 1389.90 | 1387.45 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-07-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 09:50:00 | 1366.45 | 1372.40 | 0.00 | ORB-short ORB[1367.30,1378.35] vol=2.6x ATR=3.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 10:25:00 | 1360.80 | 1369.62 | 0.00 | T1 1.5R @ 1360.80 |
| Stop hit — per-position SL triggered | 2024-07-05 10:35:00 | 1366.45 | 1369.09 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 09:40:00 | 1396.90 | 1386.82 | 0.00 | ORB-long ORB[1372.30,1389.15] vol=2.9x ATR=4.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-08 10:10:00 | 1404.35 | 1393.74 | 0.00 | T1 1.5R @ 1404.35 |
| Target hit | 2024-07-08 11:20:00 | 1417.15 | 1417.69 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — SELL (started 2024-07-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 10:45:00 | 1440.70 | 1445.24 | 0.00 | ORB-short ORB[1442.95,1452.00] vol=2.0x ATR=3.34 |
| Stop hit — per-position SL triggered | 2024-07-12 10:50:00 | 1444.04 | 1445.18 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-15 09:50:00 | 1441.90 | 1448.47 | 0.00 | ORB-short ORB[1444.45,1459.70] vol=2.6x ATR=5.37 |
| Stop hit — per-position SL triggered | 2024-07-15 10:15:00 | 1447.27 | 1446.27 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-16 11:15:00 | 1435.80 | 1439.76 | 0.00 | ORB-short ORB[1439.95,1451.85] vol=2.6x ATR=2.22 |
| Stop hit — per-position SL triggered | 2024-07-16 11:35:00 | 1438.02 | 1439.34 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-18 11:05:00 | 1457.00 | 1450.87 | 0.00 | ORB-long ORB[1440.00,1454.00] vol=2.3x ATR=3.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 11:25:00 | 1462.60 | 1452.18 | 0.00 | T1 1.5R @ 1462.60 |
| Stop hit — per-position SL triggered | 2024-07-18 11:30:00 | 1457.00 | 1452.35 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 11:05:00 | 1485.75 | 1474.33 | 0.00 | ORB-long ORB[1465.50,1481.85] vol=4.0x ATR=5.30 |
| Stop hit — per-position SL triggered | 2024-07-23 11:15:00 | 1480.45 | 1475.27 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 10:35:00 | 1461.35 | 1457.92 | 0.00 | ORB-long ORB[1443.15,1458.95] vol=1.7x ATR=3.38 |
| Stop hit — per-position SL triggered | 2024-07-25 11:30:00 | 1457.97 | 1458.54 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-26 11:05:00 | 1465.40 | 1471.35 | 0.00 | ORB-short ORB[1469.10,1480.10] vol=2.6x ATR=3.27 |
| Stop hit — per-position SL triggered | 2024-07-26 12:00:00 | 1468.67 | 1469.27 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 10:15:00 | 1458.70 | 1451.05 | 0.00 | ORB-long ORB[1436.25,1445.55] vol=4.9x ATR=3.96 |
| Stop hit — per-position SL triggered | 2024-08-01 10:40:00 | 1454.74 | 1452.41 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-08-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-12 09:45:00 | 1430.15 | 1432.57 | 0.00 | ORB-short ORB[1430.65,1445.00] vol=2.1x ATR=4.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 10:00:00 | 1423.70 | 1429.79 | 0.00 | T1 1.5R @ 1423.70 |
| Target hit | 2024-08-12 15:20:00 | 1390.60 | 1403.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — SELL (started 2024-08-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-19 10:45:00 | 1393.00 | 1393.69 | 0.00 | ORB-short ORB[1399.15,1413.55] vol=1.5x ATR=3.17 |
| Stop hit — per-position SL triggered | 2024-08-19 10:50:00 | 1396.17 | 1393.78 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 10:05:00 | 1399.85 | 1401.97 | 0.00 | ORB-short ORB[1400.00,1412.90] vol=3.6x ATR=2.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 10:25:00 | 1395.55 | 1401.03 | 0.00 | T1 1.5R @ 1395.55 |
| Target hit | 2024-08-20 13:55:00 | 1388.15 | 1387.36 | 0.00 | Trail-exit close>VWAP |

### Cycle 23 — BUY (started 2024-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 11:15:00 | 1414.95 | 1407.81 | 0.00 | ORB-long ORB[1391.50,1404.00] vol=2.7x ATR=3.38 |
| Stop hit — per-position SL triggered | 2024-08-22 12:15:00 | 1411.57 | 1409.52 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 11:00:00 | 1428.70 | 1433.73 | 0.00 | ORB-short ORB[1430.65,1446.00] vol=3.0x ATR=3.80 |
| Stop hit — per-position SL triggered | 2024-08-23 14:00:00 | 1432.50 | 1431.90 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 10:55:00 | 1447.75 | 1437.65 | 0.00 | ORB-long ORB[1422.10,1438.00] vol=3.4x ATR=3.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 11:20:00 | 1453.06 | 1442.40 | 0.00 | T1 1.5R @ 1453.06 |
| Stop hit — per-position SL triggered | 2024-08-26 12:00:00 | 1447.75 | 1446.10 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-08-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 10:50:00 | 1441.20 | 1449.10 | 0.00 | ORB-short ORB[1450.00,1459.65] vol=2.2x ATR=3.28 |
| Stop hit — per-position SL triggered | 2024-08-27 10:55:00 | 1444.48 | 1448.96 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-08-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:50:00 | 1458.40 | 1467.29 | 0.00 | ORB-short ORB[1464.60,1484.70] vol=2.2x ATR=3.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 11:00:00 | 1452.83 | 1465.75 | 0.00 | T1 1.5R @ 1452.83 |
| Stop hit — per-position SL triggered | 2024-08-29 11:40:00 | 1458.40 | 1463.91 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:35:00 | 1476.65 | 1472.67 | 0.00 | ORB-long ORB[1467.05,1476.55] vol=1.9x ATR=2.96 |
| Stop hit — per-position SL triggered | 2024-09-03 09:50:00 | 1473.69 | 1474.59 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-09-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 10:45:00 | 1460.00 | 1467.11 | 0.00 | ORB-short ORB[1470.25,1483.95] vol=1.5x ATR=3.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 12:40:00 | 1455.00 | 1462.14 | 0.00 | T1 1.5R @ 1455.00 |
| Stop hit — per-position SL triggered | 2024-09-05 15:00:00 | 1460.00 | 1460.32 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 09:55:00 | 1530.80 | 1522.47 | 0.00 | ORB-long ORB[1505.50,1526.70] vol=2.2x ATR=6.30 |
| Stop hit — per-position SL triggered | 2024-09-12 10:45:00 | 1524.50 | 1525.82 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 10:50:00 | 1449.30 | 1450.25 | 0.00 | ORB-short ORB[1449.55,1464.65] vol=2.5x ATR=3.82 |
| Stop hit — per-position SL triggered | 2024-09-18 10:55:00 | 1453.12 | 1450.36 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 09:55:00 | 1463.40 | 1457.35 | 0.00 | ORB-long ORB[1448.25,1459.85] vol=2.4x ATR=4.12 |
| Stop hit — per-position SL triggered | 2024-09-20 10:00:00 | 1459.28 | 1457.88 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-09-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 10:05:00 | 1416.05 | 1430.74 | 0.00 | ORB-short ORB[1433.80,1446.95] vol=1.5x ATR=4.27 |
| Stop hit — per-position SL triggered | 2024-09-25 10:15:00 | 1420.32 | 1429.00 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-09-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 11:10:00 | 1419.30 | 1426.39 | 0.00 | ORB-short ORB[1422.40,1432.00] vol=1.9x ATR=2.65 |
| Stop hit — per-position SL triggered | 2024-09-26 11:30:00 | 1421.95 | 1425.93 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-10-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 09:35:00 | 1340.50 | 1347.12 | 0.00 | ORB-short ORB[1343.25,1354.55] vol=3.0x ATR=5.13 |
| Stop hit — per-position SL triggered | 2024-10-07 09:40:00 | 1345.63 | 1346.59 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-10-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 10:40:00 | 1318.40 | 1311.09 | 0.00 | ORB-long ORB[1303.65,1316.70] vol=5.8x ATR=5.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-09 11:05:00 | 1326.16 | 1315.18 | 0.00 | T1 1.5R @ 1326.16 |
| Target hit | 2024-10-09 15:20:00 | 1333.10 | 1327.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2024-10-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 11:10:00 | 1310.15 | 1319.00 | 0.00 | ORB-short ORB[1311.00,1327.80] vol=1.9x ATR=3.49 |
| Stop hit — per-position SL triggered | 2024-10-14 11:20:00 | 1313.64 | 1318.88 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-10-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 09:40:00 | 1329.10 | 1325.82 | 0.00 | ORB-long ORB[1314.00,1327.05] vol=3.4x ATR=3.34 |
| Stop hit — per-position SL triggered | 2024-10-15 09:55:00 | 1325.76 | 1327.01 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-10-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 10:30:00 | 1354.00 | 1345.57 | 0.00 | ORB-long ORB[1338.05,1352.45] vol=1.7x ATR=4.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 10:40:00 | 1360.36 | 1349.75 | 0.00 | T1 1.5R @ 1360.36 |
| Stop hit — per-position SL triggered | 2024-10-16 11:05:00 | 1354.00 | 1352.47 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-10-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 10:40:00 | 1339.00 | 1351.22 | 0.00 | ORB-short ORB[1351.00,1364.85] vol=1.9x ATR=3.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 12:20:00 | 1333.33 | 1347.05 | 0.00 | T1 1.5R @ 1333.33 |
| Stop hit — per-position SL triggered | 2024-10-17 13:00:00 | 1339.00 | 1345.55 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-10-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 11:15:00 | 1321.05 | 1328.61 | 0.00 | ORB-short ORB[1329.40,1342.70] vol=1.7x ATR=3.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 12:00:00 | 1316.23 | 1327.73 | 0.00 | T1 1.5R @ 1316.23 |
| Target hit | 2024-10-21 15:20:00 | 1314.55 | 1319.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — SELL (started 2024-10-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:55:00 | 1318.20 | 1323.48 | 0.00 | ORB-short ORB[1320.00,1331.95] vol=4.0x ATR=4.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 12:10:00 | 1312.20 | 1322.45 | 0.00 | T1 1.5R @ 1312.20 |
| Target hit | 2024-10-22 15:20:00 | 1301.15 | 1311.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — SELL (started 2024-10-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-23 10:05:00 | 1288.85 | 1292.93 | 0.00 | ORB-short ORB[1290.30,1306.00] vol=1.6x ATR=4.45 |
| Stop hit — per-position SL triggered | 2024-10-23 10:15:00 | 1293.30 | 1292.68 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-10-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 10:55:00 | 1274.20 | 1278.56 | 0.00 | ORB-short ORB[1281.25,1298.05] vol=5.5x ATR=3.36 |
| Stop hit — per-position SL triggered | 2024-10-29 11:10:00 | 1277.56 | 1278.32 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-10-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 11:00:00 | 1289.70 | 1279.05 | 0.00 | ORB-long ORB[1268.05,1286.05] vol=1.8x ATR=3.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 11:10:00 | 1294.88 | 1279.97 | 0.00 | T1 1.5R @ 1294.88 |
| Target hit | 2024-10-30 15:20:00 | 1300.00 | 1290.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — SELL (started 2024-11-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 10:35:00 | 1196.00 | 1211.03 | 0.00 | ORB-short ORB[1205.00,1222.25] vol=2.6x ATR=3.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 10:55:00 | 1190.61 | 1210.02 | 0.00 | T1 1.5R @ 1190.61 |
| Stop hit — per-position SL triggered | 2024-11-12 11:20:00 | 1196.00 | 1206.45 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-11-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-21 10:50:00 | 1190.45 | 1184.05 | 0.00 | ORB-long ORB[1174.70,1186.00] vol=11.9x ATR=3.92 |
| Stop hit — per-position SL triggered | 2024-11-21 10:55:00 | 1186.53 | 1184.28 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-11-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-27 10:55:00 | 1242.65 | 1248.49 | 0.00 | ORB-short ORB[1245.00,1257.45] vol=2.4x ATR=3.38 |
| Stop hit — per-position SL triggered | 2024-11-27 11:00:00 | 1246.03 | 1248.26 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-12-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 11:10:00 | 1243.55 | 1248.25 | 0.00 | ORB-short ORB[1246.95,1255.00] vol=4.8x ATR=2.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 11:40:00 | 1239.82 | 1247.06 | 0.00 | T1 1.5R @ 1239.82 |
| Stop hit — per-position SL triggered | 2024-12-06 12:00:00 | 1243.55 | 1246.66 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 10:15:00 | 1123.70 | 1126.48 | 0.00 | ORB-short ORB[1133.85,1142.75] vol=6.6x ATR=2.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 10:40:00 | 1120.00 | 1124.36 | 0.00 | T1 1.5R @ 1120.00 |
| Stop hit — per-position SL triggered | 2024-12-12 11:00:00 | 1123.70 | 1122.78 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-12-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:30:00 | 1099.65 | 1103.70 | 0.00 | ORB-short ORB[1105.80,1116.50] vol=3.9x ATR=2.75 |
| Stop hit — per-position SL triggered | 2024-12-13 10:35:00 | 1102.40 | 1103.34 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 10:15:00 | 1113.30 | 1115.75 | 0.00 | ORB-short ORB[1114.15,1128.00] vol=6.7x ATR=3.22 |
| Stop hit — per-position SL triggered | 2024-12-16 10:30:00 | 1116.52 | 1115.61 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-12-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 11:00:00 | 1102.65 | 1103.17 | 0.00 | ORB-short ORB[1105.80,1115.25] vol=1.9x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 11:25:00 | 1099.47 | 1102.76 | 0.00 | T1 1.5R @ 1099.47 |
| Target hit | 2024-12-17 15:20:00 | 1088.90 | 1095.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — SELL (started 2024-12-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-19 10:45:00 | 1078.30 | 1080.73 | 0.00 | ORB-short ORB[1078.80,1088.00] vol=5.4x ATR=2.63 |
| Stop hit — per-position SL triggered | 2024-12-19 12:05:00 | 1080.93 | 1079.88 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-12-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-23 11:10:00 | 1069.30 | 1071.05 | 0.00 | ORB-short ORB[1069.45,1074.95] vol=1.8x ATR=2.45 |
| Stop hit — per-position SL triggered | 2024-12-23 11:30:00 | 1071.75 | 1071.66 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-12-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:00:00 | 1071.15 | 1073.61 | 0.00 | ORB-short ORB[1072.00,1083.15] vol=3.4x ATR=2.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 11:30:00 | 1067.41 | 1071.95 | 0.00 | T1 1.5R @ 1067.41 |
| Stop hit — per-position SL triggered | 2024-12-26 11:55:00 | 1071.15 | 1071.61 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-12-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 09:55:00 | 1070.60 | 1073.27 | 0.00 | ORB-short ORB[1072.00,1079.85] vol=1.6x ATR=2.44 |
| Stop hit — per-position SL triggered | 2024-12-27 10:10:00 | 1073.04 | 1072.93 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-12-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-30 11:00:00 | 1066.40 | 1068.17 | 0.00 | ORB-short ORB[1067.00,1073.05] vol=2.2x ATR=2.19 |
| Stop hit — per-position SL triggered | 2024-12-30 11:10:00 | 1068.59 | 1068.13 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-01-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 09:55:00 | 1091.00 | 1092.58 | 0.00 | ORB-short ORB[1093.35,1101.00] vol=14.4x ATR=2.80 |
| Stop hit — per-position SL triggered | 2025-01-03 10:15:00 | 1093.80 | 1092.56 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-01-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-06 11:10:00 | 1128.00 | 1122.87 | 0.00 | ORB-long ORB[1112.00,1127.75] vol=1.6x ATR=3.90 |
| Stop hit — per-position SL triggered | 2025-01-06 11:20:00 | 1124.10 | 1123.12 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-01-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 11:10:00 | 1144.95 | 1147.10 | 0.00 | ORB-short ORB[1145.00,1154.60] vol=2.4x ATR=3.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 11:45:00 | 1140.15 | 1146.51 | 0.00 | T1 1.5R @ 1140.15 |
| Stop hit — per-position SL triggered | 2025-01-08 12:50:00 | 1144.95 | 1144.22 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-01-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 09:55:00 | 1175.85 | 1164.76 | 0.00 | ORB-long ORB[1150.40,1162.50] vol=2.5x ATR=4.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 10:15:00 | 1182.92 | 1172.20 | 0.00 | T1 1.5R @ 1182.92 |
| Target hit | 2025-01-09 12:25:00 | 1182.70 | 1184.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 63 — SELL (started 2025-01-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 10:00:00 | 1169.00 | 1173.39 | 0.00 | ORB-short ORB[1172.05,1189.55] vol=1.8x ATR=3.82 |
| Stop hit — per-position SL triggered | 2025-01-10 10:05:00 | 1172.82 | 1173.32 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-01-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-13 09:40:00 | 1178.25 | 1171.89 | 0.00 | ORB-long ORB[1162.05,1173.75] vol=2.3x ATR=3.49 |
| Stop hit — per-position SL triggered | 2025-01-13 09:50:00 | 1174.76 | 1173.03 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-01-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-14 09:30:00 | 1151.00 | 1154.77 | 0.00 | ORB-short ORB[1153.30,1164.75] vol=2.4x ATR=3.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-14 09:45:00 | 1145.01 | 1151.38 | 0.00 | T1 1.5R @ 1145.01 |
| Target hit | 2025-01-14 11:10:00 | 1147.90 | 1144.33 | 0.00 | Trail-exit close>VWAP |

### Cycle 66 — SELL (started 2025-01-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 09:50:00 | 1134.35 | 1137.33 | 0.00 | ORB-short ORB[1136.20,1142.95] vol=4.0x ATR=3.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-15 12:50:00 | 1128.60 | 1134.36 | 0.00 | T1 1.5R @ 1128.60 |
| Target hit | 2025-01-15 15:20:00 | 1130.35 | 1132.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 67 — BUY (started 2025-01-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 09:55:00 | 1163.05 | 1159.59 | 0.00 | ORB-long ORB[1149.00,1159.05] vol=1.6x ATR=3.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 10:05:00 | 1167.66 | 1160.81 | 0.00 | T1 1.5R @ 1167.66 |
| Target hit | 2025-01-17 15:20:00 | 1189.45 | 1185.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 68 — SELL (started 2025-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-20 11:15:00 | 1173.50 | 1178.77 | 0.00 | ORB-short ORB[1181.10,1193.95] vol=1.7x ATR=2.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 11:20:00 | 1169.08 | 1178.44 | 0.00 | T1 1.5R @ 1169.08 |
| Stop hit — per-position SL triggered | 2025-01-20 11:30:00 | 1173.50 | 1178.33 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-01-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 10:55:00 | 1135.50 | 1136.80 | 0.00 | ORB-short ORB[1138.50,1147.95] vol=1.6x ATR=2.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 12:05:00 | 1131.05 | 1136.19 | 0.00 | T1 1.5R @ 1131.05 |
| Stop hit — per-position SL triggered | 2025-01-24 13:55:00 | 1135.50 | 1134.31 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-01-28 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-28 10:00:00 | 1130.00 | 1122.30 | 0.00 | ORB-long ORB[1112.90,1126.65] vol=3.2x ATR=4.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 10:10:00 | 1136.79 | 1124.23 | 0.00 | T1 1.5R @ 1136.79 |
| Stop hit — per-position SL triggered | 2025-01-28 10:40:00 | 1130.00 | 1126.61 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-02-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 11:00:00 | 1137.00 | 1134.43 | 0.00 | ORB-long ORB[1118.55,1131.35] vol=1.9x ATR=3.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 11:15:00 | 1142.15 | 1134.64 | 0.00 | T1 1.5R @ 1142.15 |
| Stop hit — per-position SL triggered | 2025-02-01 11:40:00 | 1137.00 | 1135.18 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2025-02-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-07 09:35:00 | 1103.10 | 1107.79 | 0.00 | ORB-short ORB[1105.75,1122.05] vol=1.8x ATR=2.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-07 10:00:00 | 1098.73 | 1102.51 | 0.00 | T1 1.5R @ 1098.73 |
| Stop hit — per-position SL triggered | 2025-02-07 10:30:00 | 1103.10 | 1101.88 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2025-02-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-12 10:05:00 | 1056.95 | 1060.96 | 0.00 | ORB-short ORB[1060.55,1075.55] vol=1.5x ATR=3.00 |
| Stop hit — per-position SL triggered | 2025-02-12 10:35:00 | 1059.95 | 1060.36 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2025-02-14 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-14 10:20:00 | 1072.95 | 1065.82 | 0.00 | ORB-long ORB[1052.10,1061.80] vol=1.5x ATR=3.16 |
| Stop hit — per-position SL triggered | 2025-02-14 10:30:00 | 1069.79 | 1067.63 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2025-03-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 11:00:00 | 1059.50 | 1053.05 | 0.00 | ORB-long ORB[1038.30,1053.50] vol=1.5x ATR=3.78 |
| Stop hit — per-position SL triggered | 2025-03-10 11:20:00 | 1055.72 | 1054.43 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2025-03-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-13 10:45:00 | 1030.45 | 1034.08 | 0.00 | ORB-short ORB[1034.80,1045.00] vol=1.8x ATR=2.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-13 10:55:00 | 1026.44 | 1033.33 | 0.00 | T1 1.5R @ 1026.44 |
| Stop hit — per-position SL triggered | 2025-03-13 11:00:00 | 1030.45 | 1033.12 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2025-03-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:45:00 | 1109.70 | 1102.83 | 0.00 | ORB-long ORB[1094.00,1105.95] vol=2.0x ATR=3.30 |
| Stop hit — per-position SL triggered | 2025-03-21 09:50:00 | 1106.40 | 1104.07 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2025-03-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-25 09:40:00 | 1116.65 | 1107.77 | 0.00 | ORB-long ORB[1097.60,1111.85] vol=2.6x ATR=4.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-25 09:55:00 | 1122.68 | 1109.98 | 0.00 | T1 1.5R @ 1122.68 |
| Target hit | 2025-03-25 13:50:00 | 1126.45 | 1126.69 | 0.00 | Trail-exit close<VWAP |

### Cycle 79 — SELL (started 2025-04-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-04 10:25:00 | 1144.30 | 1157.76 | 0.00 | ORB-short ORB[1159.60,1173.45] vol=2.8x ATR=4.18 |
| Stop hit — per-position SL triggered | 2025-04-04 10:30:00 | 1148.48 | 1156.95 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2025-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-09 10:15:00 | 1216.45 | 1210.14 | 0.00 | ORB-long ORB[1196.25,1211.20] vol=4.7x ATR=3.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-09 10:30:00 | 1222.38 | 1211.98 | 0.00 | T1 1.5R @ 1222.38 |
| Target hit | 2025-04-09 15:20:00 | 1240.95 | 1232.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 81 — SELL (started 2025-04-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-15 11:00:00 | 1217.50 | 1226.61 | 0.00 | ORB-short ORB[1225.70,1239.00] vol=2.0x ATR=3.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-15 11:20:00 | 1212.28 | 1224.51 | 0.00 | T1 1.5R @ 1212.28 |
| Target hit | 2025-04-15 14:50:00 | 1214.30 | 1213.09 | 0.00 | Trail-exit close>VWAP |

### Cycle 82 — BUY (started 2025-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 11:15:00 | 1240.00 | 1227.43 | 0.00 | ORB-long ORB[1211.60,1225.00] vol=1.8x ATR=2.98 |
| Stop hit — per-position SL triggered | 2025-04-16 11:30:00 | 1237.02 | 1228.55 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2025-04-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 09:55:00 | 1234.00 | 1227.41 | 0.00 | ORB-long ORB[1214.40,1232.20] vol=2.1x ATR=4.37 |
| Stop hit — per-position SL triggered | 2025-04-17 10:05:00 | 1229.63 | 1228.49 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2025-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 09:30:00 | 1230.80 | 1225.15 | 0.00 | ORB-long ORB[1214.00,1229.70] vol=1.7x ATR=3.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-22 09:35:00 | 1235.84 | 1228.28 | 0.00 | T1 1.5R @ 1235.84 |
| Target hit | 2025-04-22 12:50:00 | 1235.70 | 1238.25 | 0.00 | Trail-exit close<VWAP |

### Cycle 85 — BUY (started 2025-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 09:35:00 | 1249.40 | 1242.81 | 0.00 | ORB-long ORB[1231.40,1245.50] vol=1.6x ATR=4.24 |
| Stop hit — per-position SL triggered | 2025-04-23 09:45:00 | 1245.16 | 1244.19 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2025-04-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-24 10:35:00 | 1253.20 | 1265.18 | 0.00 | ORB-short ORB[1260.80,1275.00] vol=1.7x ATR=5.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-24 10:55:00 | 1245.60 | 1262.58 | 0.00 | T1 1.5R @ 1245.60 |
| Target hit | 2025-04-24 12:40:00 | 1251.00 | 1249.29 | 0.00 | Trail-exit close>VWAP |

### Cycle 87 — SELL (started 2025-04-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-28 09:40:00 | 1252.20 | 1258.14 | 0.00 | ORB-short ORB[1255.10,1270.60] vol=1.7x ATR=4.13 |
| Stop hit — per-position SL triggered | 2025-04-28 09:45:00 | 1256.33 | 1257.90 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2025-04-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-29 11:05:00 | 1269.10 | 1257.79 | 0.00 | ORB-long ORB[1251.00,1267.00] vol=3.0x ATR=4.18 |
| Stop hit — per-position SL triggered | 2025-04-29 11:45:00 | 1264.92 | 1259.78 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2025-05-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-02 11:10:00 | 1272.70 | 1270.11 | 0.00 | ORB-long ORB[1255.00,1269.40] vol=1.5x ATR=3.65 |
| Stop hit — per-position SL triggered | 2025-05-02 11:30:00 | 1269.05 | 1270.14 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2025-05-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 09:40:00 | 1279.20 | 1270.45 | 0.00 | ORB-long ORB[1260.90,1273.50] vol=1.6x ATR=4.87 |
| Stop hit — per-position SL triggered | 2025-05-05 09:45:00 | 1274.33 | 1270.54 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-13 11:00:00 | 1334.90 | 2024-05-13 11:50:00 | 1342.91 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-05-13 11:00:00 | 1334.90 | 2024-05-13 15:00:00 | 1334.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-22 09:35:00 | 1304.30 | 2024-05-22 09:40:00 | 1300.42 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-05-31 10:20:00 | 1270.45 | 2024-05-31 10:25:00 | 1274.78 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-06-07 10:10:00 | 1420.30 | 2024-06-07 10:15:00 | 1414.75 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-06-10 09:40:00 | 1426.15 | 2024-06-10 09:55:00 | 1415.79 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2024-06-10 09:40:00 | 1426.15 | 2024-06-10 11:15:00 | 1426.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-18 10:05:00 | 1399.00 | 2024-06-18 10:10:00 | 1394.41 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-06-19 10:45:00 | 1389.70 | 2024-06-19 10:50:00 | 1393.85 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-06-26 10:05:00 | 1388.20 | 2024-06-26 10:10:00 | 1384.37 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-01 10:30:00 | 1389.90 | 2024-07-01 10:40:00 | 1395.49 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-07-01 10:30:00 | 1389.90 | 2024-07-01 12:30:00 | 1389.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-05 09:50:00 | 1366.45 | 2024-07-05 10:25:00 | 1360.80 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-07-05 09:50:00 | 1366.45 | 2024-07-05 10:35:00 | 1366.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-08 09:40:00 | 1396.90 | 2024-07-08 10:10:00 | 1404.35 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-07-08 09:40:00 | 1396.90 | 2024-07-08 11:20:00 | 1417.15 | TARGET_HIT | 0.50 | 1.45% |
| SELL | retest1 | 2024-07-12 10:45:00 | 1440.70 | 2024-07-12 10:50:00 | 1444.04 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-07-15 09:50:00 | 1441.90 | 2024-07-15 10:15:00 | 1447.27 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-07-16 11:15:00 | 1435.80 | 2024-07-16 11:35:00 | 1438.02 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2024-07-18 11:05:00 | 1457.00 | 2024-07-18 11:25:00 | 1462.60 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-07-18 11:05:00 | 1457.00 | 2024-07-18 11:30:00 | 1457.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-23 11:05:00 | 1485.75 | 2024-07-23 11:15:00 | 1480.45 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-07-25 10:35:00 | 1461.35 | 2024-07-25 11:30:00 | 1457.97 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-07-26 11:05:00 | 1465.40 | 2024-07-26 12:00:00 | 1468.67 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-08-01 10:15:00 | 1458.70 | 2024-08-01 10:40:00 | 1454.74 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-08-12 09:45:00 | 1430.15 | 2024-08-12 10:00:00 | 1423.70 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-08-12 09:45:00 | 1430.15 | 2024-08-12 15:20:00 | 1390.60 | TARGET_HIT | 0.50 | 2.77% |
| SELL | retest1 | 2024-08-19 10:45:00 | 1393.00 | 2024-08-19 10:50:00 | 1396.17 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-08-20 10:05:00 | 1399.85 | 2024-08-20 10:25:00 | 1395.55 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-08-20 10:05:00 | 1399.85 | 2024-08-20 13:55:00 | 1388.15 | TARGET_HIT | 0.50 | 0.84% |
| BUY | retest1 | 2024-08-22 11:15:00 | 1414.95 | 2024-08-22 12:15:00 | 1411.57 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-08-23 11:00:00 | 1428.70 | 2024-08-23 14:00:00 | 1432.50 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-08-26 10:55:00 | 1447.75 | 2024-08-26 11:20:00 | 1453.06 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-08-26 10:55:00 | 1447.75 | 2024-08-26 12:00:00 | 1447.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-27 10:50:00 | 1441.20 | 2024-08-27 10:55:00 | 1444.48 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-08-29 10:50:00 | 1458.40 | 2024-08-29 11:00:00 | 1452.83 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-08-29 10:50:00 | 1458.40 | 2024-08-29 11:40:00 | 1458.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-03 09:35:00 | 1476.65 | 2024-09-03 09:50:00 | 1473.69 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-09-05 10:45:00 | 1460.00 | 2024-09-05 12:40:00 | 1455.00 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-09-05 10:45:00 | 1460.00 | 2024-09-05 15:00:00 | 1460.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-12 09:55:00 | 1530.80 | 2024-09-12 10:45:00 | 1524.50 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-09-18 10:50:00 | 1449.30 | 2024-09-18 10:55:00 | 1453.12 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-09-20 09:55:00 | 1463.40 | 2024-09-20 10:00:00 | 1459.28 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-09-25 10:05:00 | 1416.05 | 2024-09-25 10:15:00 | 1420.32 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-09-26 11:10:00 | 1419.30 | 2024-09-26 11:30:00 | 1421.95 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-10-07 09:35:00 | 1340.50 | 2024-10-07 09:40:00 | 1345.63 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-10-09 10:40:00 | 1318.40 | 2024-10-09 11:05:00 | 1326.16 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-10-09 10:40:00 | 1318.40 | 2024-10-09 15:20:00 | 1333.10 | TARGET_HIT | 0.50 | 1.11% |
| SELL | retest1 | 2024-10-14 11:10:00 | 1310.15 | 2024-10-14 11:20:00 | 1313.64 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-10-15 09:40:00 | 1329.10 | 2024-10-15 09:55:00 | 1325.76 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-10-16 10:30:00 | 1354.00 | 2024-10-16 10:40:00 | 1360.36 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-10-16 10:30:00 | 1354.00 | 2024-10-16 11:05:00 | 1354.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-17 10:40:00 | 1339.00 | 2024-10-17 12:20:00 | 1333.33 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-10-17 10:40:00 | 1339.00 | 2024-10-17 13:00:00 | 1339.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-21 11:15:00 | 1321.05 | 2024-10-21 12:00:00 | 1316.23 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-10-21 11:15:00 | 1321.05 | 2024-10-21 15:20:00 | 1314.55 | TARGET_HIT | 0.50 | 0.49% |
| SELL | retest1 | 2024-10-22 10:55:00 | 1318.20 | 2024-10-22 12:10:00 | 1312.20 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-10-22 10:55:00 | 1318.20 | 2024-10-22 15:20:00 | 1301.15 | TARGET_HIT | 0.50 | 1.29% |
| SELL | retest1 | 2024-10-23 10:05:00 | 1288.85 | 2024-10-23 10:15:00 | 1293.30 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-10-29 10:55:00 | 1274.20 | 2024-10-29 11:10:00 | 1277.56 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-10-30 11:00:00 | 1289.70 | 2024-10-30 11:10:00 | 1294.88 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-10-30 11:00:00 | 1289.70 | 2024-10-30 15:20:00 | 1300.00 | TARGET_HIT | 0.50 | 0.80% |
| SELL | retest1 | 2024-11-12 10:35:00 | 1196.00 | 2024-11-12 10:55:00 | 1190.61 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-11-12 10:35:00 | 1196.00 | 2024-11-12 11:20:00 | 1196.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-21 10:50:00 | 1190.45 | 2024-11-21 10:55:00 | 1186.53 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-11-27 10:55:00 | 1242.65 | 2024-11-27 11:00:00 | 1246.03 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-12-06 11:10:00 | 1243.55 | 2024-12-06 11:40:00 | 1239.82 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-12-06 11:10:00 | 1243.55 | 2024-12-06 12:00:00 | 1243.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-12 10:15:00 | 1123.70 | 2024-12-12 10:40:00 | 1120.00 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-12-12 10:15:00 | 1123.70 | 2024-12-12 11:00:00 | 1123.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-13 10:30:00 | 1099.65 | 2024-12-13 10:35:00 | 1102.40 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-12-16 10:15:00 | 1113.30 | 2024-12-16 10:30:00 | 1116.52 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-12-17 11:00:00 | 1102.65 | 2024-12-17 11:25:00 | 1099.47 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-12-17 11:00:00 | 1102.65 | 2024-12-17 15:20:00 | 1088.90 | TARGET_HIT | 0.50 | 1.25% |
| SELL | retest1 | 2024-12-19 10:45:00 | 1078.30 | 2024-12-19 12:05:00 | 1080.93 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-12-23 11:10:00 | 1069.30 | 2024-12-23 11:30:00 | 1071.75 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-12-26 10:00:00 | 1071.15 | 2024-12-26 11:30:00 | 1067.41 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-12-26 10:00:00 | 1071.15 | 2024-12-26 11:55:00 | 1071.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-27 09:55:00 | 1070.60 | 2024-12-27 10:10:00 | 1073.04 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-12-30 11:00:00 | 1066.40 | 2024-12-30 11:10:00 | 1068.59 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-01-03 09:55:00 | 1091.00 | 2025-01-03 10:15:00 | 1093.80 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-01-06 11:10:00 | 1128.00 | 2025-01-06 11:20:00 | 1124.10 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-01-08 11:10:00 | 1144.95 | 2025-01-08 11:45:00 | 1140.15 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-01-08 11:10:00 | 1144.95 | 2025-01-08 12:50:00 | 1144.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-09 09:55:00 | 1175.85 | 2025-01-09 10:15:00 | 1182.92 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-01-09 09:55:00 | 1175.85 | 2025-01-09 12:25:00 | 1182.70 | TARGET_HIT | 0.50 | 0.58% |
| SELL | retest1 | 2025-01-10 10:00:00 | 1169.00 | 2025-01-10 10:05:00 | 1172.82 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-01-13 09:40:00 | 1178.25 | 2025-01-13 09:50:00 | 1174.76 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-01-14 09:30:00 | 1151.00 | 2025-01-14 09:45:00 | 1145.01 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-01-14 09:30:00 | 1151.00 | 2025-01-14 11:10:00 | 1147.90 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2025-01-15 09:50:00 | 1134.35 | 2025-01-15 12:50:00 | 1128.60 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-01-15 09:50:00 | 1134.35 | 2025-01-15 15:20:00 | 1130.35 | TARGET_HIT | 0.50 | 0.35% |
| BUY | retest1 | 2025-01-17 09:55:00 | 1163.05 | 2025-01-17 10:05:00 | 1167.66 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-01-17 09:55:00 | 1163.05 | 2025-01-17 15:20:00 | 1189.45 | TARGET_HIT | 0.50 | 2.27% |
| SELL | retest1 | 2025-01-20 11:15:00 | 1173.50 | 2025-01-20 11:20:00 | 1169.08 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-01-20 11:15:00 | 1173.50 | 2025-01-20 11:30:00 | 1173.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-24 10:55:00 | 1135.50 | 2025-01-24 12:05:00 | 1131.05 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-01-24 10:55:00 | 1135.50 | 2025-01-24 13:55:00 | 1135.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-28 10:00:00 | 1130.00 | 2025-01-28 10:10:00 | 1136.79 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-01-28 10:00:00 | 1130.00 | 2025-01-28 10:40:00 | 1130.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-01 11:00:00 | 1137.00 | 2025-02-01 11:15:00 | 1142.15 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-02-01 11:00:00 | 1137.00 | 2025-02-01 11:40:00 | 1137.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-07 09:35:00 | 1103.10 | 2025-02-07 10:00:00 | 1098.73 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-02-07 09:35:00 | 1103.10 | 2025-02-07 10:30:00 | 1103.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-12 10:05:00 | 1056.95 | 2025-02-12 10:35:00 | 1059.95 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-02-14 10:20:00 | 1072.95 | 2025-02-14 10:30:00 | 1069.79 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-03-10 11:00:00 | 1059.50 | 2025-03-10 11:20:00 | 1055.72 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-03-13 10:45:00 | 1030.45 | 2025-03-13 10:55:00 | 1026.44 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-03-13 10:45:00 | 1030.45 | 2025-03-13 11:00:00 | 1030.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-21 09:45:00 | 1109.70 | 2025-03-21 09:50:00 | 1106.40 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-03-25 09:40:00 | 1116.65 | 2025-03-25 09:55:00 | 1122.68 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-03-25 09:40:00 | 1116.65 | 2025-03-25 13:50:00 | 1126.45 | TARGET_HIT | 0.50 | 0.88% |
| SELL | retest1 | 2025-04-04 10:25:00 | 1144.30 | 2025-04-04 10:30:00 | 1148.48 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-04-09 10:15:00 | 1216.45 | 2025-04-09 10:30:00 | 1222.38 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-04-09 10:15:00 | 1216.45 | 2025-04-09 15:20:00 | 1240.95 | TARGET_HIT | 0.50 | 2.01% |
| SELL | retest1 | 2025-04-15 11:00:00 | 1217.50 | 2025-04-15 11:20:00 | 1212.28 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-04-15 11:00:00 | 1217.50 | 2025-04-15 14:50:00 | 1214.30 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2025-04-16 11:15:00 | 1240.00 | 2025-04-16 11:30:00 | 1237.02 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-04-17 09:55:00 | 1234.00 | 2025-04-17 10:05:00 | 1229.63 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-04-22 09:30:00 | 1230.80 | 2025-04-22 09:35:00 | 1235.84 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-04-22 09:30:00 | 1230.80 | 2025-04-22 12:50:00 | 1235.70 | TARGET_HIT | 0.50 | 0.40% |
| BUY | retest1 | 2025-04-23 09:35:00 | 1249.40 | 2025-04-23 09:45:00 | 1245.16 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-04-24 10:35:00 | 1253.20 | 2025-04-24 10:55:00 | 1245.60 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2025-04-24 10:35:00 | 1253.20 | 2025-04-24 12:40:00 | 1251.00 | TARGET_HIT | 0.50 | 0.18% |
| SELL | retest1 | 2025-04-28 09:40:00 | 1252.20 | 2025-04-28 09:45:00 | 1256.33 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-04-29 11:05:00 | 1269.10 | 2025-04-29 11:45:00 | 1264.92 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-05-02 11:10:00 | 1272.70 | 2025-05-02 11:30:00 | 1269.05 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-05-05 09:40:00 | 1279.20 | 2025-05-05 09:45:00 | 1274.33 | STOP_HIT | 1.00 | -0.38% |
