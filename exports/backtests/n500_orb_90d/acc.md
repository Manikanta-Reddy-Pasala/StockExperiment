# ACC Ltd. (ACC)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1393.00
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
| ENTRY1 | 22 |
| ENTRY2 | 0 |
| PARTIAL | 12 |
| TARGET_HIT | 5 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 34 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 17
- **Target hits / Stop hits / Partials:** 5 / 17 / 12
- **Avg / median % per leg:** 0.15% / 0.12%
- **Sum % (uncompounded):** 5.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 10 | 50.0% | 3 | 10 | 7 | 0.21% | 4.2% |
| BUY @ 2nd Alert (retest1) | 20 | 10 | 50.0% | 3 | 10 | 7 | 0.21% | 4.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 7 | 50.0% | 2 | 7 | 5 | 0.07% | 1.0% |
| SELL @ 2nd Alert (retest1) | 14 | 7 | 50.0% | 2 | 7 | 5 | 0.07% | 1.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 34 | 17 | 50.0% | 5 | 17 | 12 | 0.15% | 5.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:45:00 | 1687.50 | 1681.54 | 0.00 | ORB-long ORB[1671.90,1683.70] vol=2.2x ATR=5.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 12:15:00 | 1696.17 | 1685.85 | 0.00 | T1 1.5R @ 1696.17 |
| Target hit | 2026-02-09 15:20:00 | 1706.50 | 1695.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:35:00 | 1696.10 | 1704.38 | 0.00 | ORB-short ORB[1702.20,1713.50] vol=1.7x ATR=3.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:40:00 | 1690.69 | 1702.50 | 0.00 | T1 1.5R @ 1690.69 |
| Stop hit — per-position SL triggered | 2026-02-10 10:45:00 | 1696.10 | 1702.32 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:50:00 | 1684.00 | 1689.63 | 0.00 | ORB-short ORB[1685.90,1703.90] vol=1.9x ATR=3.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:15:00 | 1679.43 | 1688.02 | 0.00 | T1 1.5R @ 1679.43 |
| Stop hit — per-position SL triggered | 2026-02-12 12:20:00 | 1684.00 | 1686.20 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:50:00 | 1642.40 | 1635.74 | 0.00 | ORB-long ORB[1625.70,1636.50] vol=1.7x ATR=3.67 |
| Stop hit — per-position SL triggered | 2026-02-17 10:05:00 | 1638.73 | 1636.41 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:35:00 | 1617.30 | 1611.04 | 0.00 | ORB-long ORB[1602.00,1614.00] vol=1.8x ATR=3.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 12:15:00 | 1623.04 | 1614.12 | 0.00 | T1 1.5R @ 1623.04 |
| Stop hit — per-position SL triggered | 2026-02-20 13:10:00 | 1617.30 | 1615.10 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:40:00 | 1629.80 | 1624.50 | 0.00 | ORB-long ORB[1619.00,1628.00] vol=4.1x ATR=3.13 |
| Stop hit — per-position SL triggered | 2026-02-25 10:45:00 | 1626.67 | 1624.86 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:40:00 | 1599.10 | 1604.15 | 0.00 | ORB-short ORB[1603.00,1617.20] vol=3.3x ATR=3.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:00:00 | 1593.85 | 1602.10 | 0.00 | T1 1.5R @ 1593.85 |
| Target hit | 2026-02-27 14:30:00 | 1595.60 | 1595.31 | 0.00 | Trail-exit close>VWAP |

### Cycle 8 — SELL (started 2026-03-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:40:00 | 1518.40 | 1521.40 | 0.00 | ORB-short ORB[1520.10,1528.40] vol=2.8x ATR=3.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:35:00 | 1512.42 | 1519.75 | 0.00 | T1 1.5R @ 1512.42 |
| Target hit | 2026-03-05 14:35:00 | 1515.20 | 1515.06 | 0.00 | Trail-exit close>VWAP |

### Cycle 9 — BUY (started 2026-03-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 09:30:00 | 1525.00 | 1521.44 | 0.00 | ORB-long ORB[1508.40,1524.30] vol=1.8x ATR=4.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:05:00 | 1532.25 | 1524.51 | 0.00 | T1 1.5R @ 1532.25 |
| Stop hit — per-position SL triggered | 2026-03-06 10:30:00 | 1525.00 | 1525.50 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:15:00 | 1477.00 | 1459.62 | 0.00 | ORB-long ORB[1450.70,1470.00] vol=1.5x ATR=5.44 |
| Stop hit — per-position SL triggered | 2026-03-10 11:10:00 | 1471.56 | 1463.44 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:50:00 | 1410.00 | 1417.02 | 0.00 | ORB-short ORB[1415.00,1430.10] vol=1.9x ATR=4.65 |
| Stop hit — per-position SL triggered | 2026-03-13 10:10:00 | 1414.65 | 1416.22 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-06 10:10:00 | 1335.80 | 1324.72 | 0.00 | ORB-long ORB[1310.10,1325.90] vol=1.8x ATR=5.62 |
| Stop hit — per-position SL triggered | 2026-04-06 11:00:00 | 1330.18 | 1328.13 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-07 11:00:00 | 1338.80 | 1348.47 | 0.00 | ORB-short ORB[1344.00,1357.90] vol=2.8x ATR=4.02 |
| Stop hit — per-position SL triggered | 2026-04-07 13:40:00 | 1342.82 | 1344.31 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:30:00 | 1397.30 | 1391.45 | 0.00 | ORB-long ORB[1376.00,1396.80] vol=2.9x ATR=6.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 09:45:00 | 1406.32 | 1394.20 | 0.00 | T1 1.5R @ 1406.32 |
| Target hit | 2026-04-08 14:25:00 | 1415.60 | 1416.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — BUY (started 2026-04-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:50:00 | 1413.90 | 1406.83 | 0.00 | ORB-long ORB[1395.60,1407.50] vol=1.8x ATR=3.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 12:10:00 | 1419.01 | 1409.35 | 0.00 | T1 1.5R @ 1419.01 |
| Target hit | 2026-04-10 14:35:00 | 1415.60 | 1417.43 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — SELL (started 2026-04-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:35:00 | 1432.00 | 1438.25 | 0.00 | ORB-short ORB[1434.90,1447.10] vol=2.0x ATR=4.02 |
| Stop hit — per-position SL triggered | 2026-04-16 10:45:00 | 1436.02 | 1437.84 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:10:00 | 1444.10 | 1437.18 | 0.00 | ORB-long ORB[1425.00,1438.60] vol=2.1x ATR=3.16 |
| Stop hit — per-position SL triggered | 2026-04-21 10:15:00 | 1440.94 | 1437.42 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-04-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 10:35:00 | 1424.80 | 1430.26 | 0.00 | ORB-short ORB[1425.10,1439.70] vol=1.6x ATR=3.18 |
| Stop hit — per-position SL triggered | 2026-04-23 12:00:00 | 1427.98 | 1428.87 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2026-04-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:00:00 | 1409.10 | 1411.81 | 0.00 | ORB-short ORB[1409.60,1429.00] vol=2.2x ATR=3.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:40:00 | 1403.13 | 1411.06 | 0.00 | T1 1.5R @ 1403.13 |
| Stop hit — per-position SL triggered | 2026-04-24 13:55:00 | 1409.10 | 1407.67 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2026-04-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:30:00 | 1429.60 | 1423.88 | 0.00 | ORB-long ORB[1416.40,1426.30] vol=2.1x ATR=4.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:45:00 | 1436.81 | 1427.10 | 0.00 | T1 1.5R @ 1436.81 |
| Stop hit — per-position SL triggered | 2026-04-27 10:05:00 | 1429.60 | 1427.81 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2026-05-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:00:00 | 1425.90 | 1413.10 | 0.00 | ORB-long ORB[1402.00,1420.20] vol=1.8x ATR=4.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 11:35:00 | 1433.33 | 1416.02 | 0.00 | T1 1.5R @ 1433.33 |
| Stop hit — per-position SL triggered | 2026-05-04 11:55:00 | 1425.90 | 1417.40 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2026-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:15:00 | 1430.00 | 1423.55 | 0.00 | ORB-long ORB[1409.20,1421.70] vol=1.8x ATR=3.98 |
| Stop hit — per-position SL triggered | 2026-05-08 10:35:00 | 1426.02 | 1425.13 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:45:00 | 1687.50 | 2026-02-09 12:15:00 | 1696.17 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-02-09 10:45:00 | 1687.50 | 2026-02-09 15:20:00 | 1706.50 | TARGET_HIT | 0.50 | 1.13% |
| SELL | retest1 | 2026-02-10 10:35:00 | 1696.10 | 2026-02-10 10:40:00 | 1690.69 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-02-10 10:35:00 | 1696.10 | 2026-02-10 10:45:00 | 1696.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-12 10:50:00 | 1684.00 | 2026-02-12 11:15:00 | 1679.43 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2026-02-12 10:50:00 | 1684.00 | 2026-02-12 12:20:00 | 1684.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 09:50:00 | 1642.40 | 2026-02-17 10:05:00 | 1638.73 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-20 10:35:00 | 1617.30 | 2026-02-20 12:15:00 | 1623.04 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-02-20 10:35:00 | 1617.30 | 2026-02-20 13:10:00 | 1617.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 10:40:00 | 1629.80 | 2026-02-25 10:45:00 | 1626.67 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-02-27 09:40:00 | 1599.10 | 2026-02-27 10:00:00 | 1593.85 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-02-27 09:40:00 | 1599.10 | 2026-02-27 14:30:00 | 1595.60 | TARGET_HIT | 0.50 | 0.22% |
| SELL | retest1 | 2026-03-05 10:40:00 | 1518.40 | 2026-03-05 11:35:00 | 1512.42 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-03-05 10:40:00 | 1518.40 | 2026-03-05 14:35:00 | 1515.20 | TARGET_HIT | 0.50 | 0.21% |
| BUY | retest1 | 2026-03-06 09:30:00 | 1525.00 | 2026-03-06 10:05:00 | 1532.25 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-03-06 09:30:00 | 1525.00 | 2026-03-06 10:30:00 | 1525.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-10 10:15:00 | 1477.00 | 2026-03-10 11:10:00 | 1471.56 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-03-13 09:50:00 | 1410.00 | 2026-03-13 10:10:00 | 1414.65 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-06 10:10:00 | 1335.80 | 2026-04-06 11:00:00 | 1330.18 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-04-07 11:00:00 | 1338.80 | 2026-04-07 13:40:00 | 1342.82 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-08 09:30:00 | 1397.30 | 2026-04-08 09:45:00 | 1406.32 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-04-08 09:30:00 | 1397.30 | 2026-04-08 14:25:00 | 1415.60 | TARGET_HIT | 0.50 | 1.31% |
| BUY | retest1 | 2026-04-10 10:50:00 | 1413.90 | 2026-04-10 12:10:00 | 1419.01 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-04-10 10:50:00 | 1413.90 | 2026-04-10 14:35:00 | 1415.60 | TARGET_HIT | 0.50 | 0.12% |
| SELL | retest1 | 2026-04-16 10:35:00 | 1432.00 | 2026-04-16 10:45:00 | 1436.02 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-21 10:10:00 | 1444.10 | 2026-04-21 10:15:00 | 1440.94 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-04-23 10:35:00 | 1424.80 | 2026-04-23 12:00:00 | 1427.98 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-04-24 10:00:00 | 1409.10 | 2026-04-24 10:40:00 | 1403.13 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-04-24 10:00:00 | 1409.10 | 2026-04-24 13:55:00 | 1409.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 09:30:00 | 1429.60 | 2026-04-27 09:45:00 | 1436.81 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-04-27 09:30:00 | 1429.60 | 2026-04-27 10:05:00 | 1429.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-04 11:00:00 | 1425.90 | 2026-05-04 11:35:00 | 1433.33 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-05-04 11:00:00 | 1425.90 | 2026-05-04 11:55:00 | 1425.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-08 10:15:00 | 1430.00 | 2026-05-08 10:35:00 | 1426.02 | STOP_HIT | 1.00 | -0.28% |
