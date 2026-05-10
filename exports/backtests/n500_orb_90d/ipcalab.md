# Ipca Laboratories Ltd. (IPCALAB)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1554.00
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
| ENTRY1 | 23 |
| ENTRY2 | 0 |
| PARTIAL | 10 |
| TARGET_HIT | 3 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 33 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 20
- **Target hits / Stop hits / Partials:** 3 / 20 / 10
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 3.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 7 | 41.2% | 2 | 10 | 5 | 0.17% | 2.9% |
| BUY @ 2nd Alert (retest1) | 17 | 7 | 41.2% | 2 | 10 | 5 | 0.17% | 2.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 16 | 6 | 37.5% | 1 | 10 | 5 | 0.06% | 1.0% |
| SELL @ 2nd Alert (retest1) | 16 | 6 | 37.5% | 1 | 10 | 5 | 0.06% | 1.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 33 | 13 | 39.4% | 3 | 20 | 10 | 0.12% | 3.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 11:10:00 | 1437.30 | 1446.80 | 0.00 | ORB-short ORB[1455.00,1467.00] vol=3.1x ATR=3.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 12:10:00 | 1431.40 | 1442.49 | 0.00 | T1 1.5R @ 1431.40 |
| Target hit | 2026-02-10 15:20:00 | 1429.00 | 1434.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:00:00 | 1481.70 | 1489.73 | 0.00 | ORB-short ORB[1488.30,1501.80] vol=1.8x ATR=4.30 |
| Stop hit — per-position SL triggered | 2026-02-18 12:25:00 | 1486.00 | 1487.14 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:30:00 | 1477.40 | 1488.92 | 0.00 | ORB-short ORB[1485.90,1501.40] vol=1.9x ATR=4.48 |
| Stop hit — per-position SL triggered | 2026-02-19 10:40:00 | 1481.88 | 1488.00 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-20 10:55:00 | 1446.20 | 1453.45 | 0.00 | ORB-short ORB[1447.00,1461.80] vol=2.2x ATR=4.49 |
| Stop hit — per-position SL triggered | 2026-02-20 11:20:00 | 1450.69 | 1452.14 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 11:10:00 | 1488.90 | 1478.69 | 0.00 | ORB-long ORB[1465.00,1483.00] vol=1.8x ATR=4.75 |
| Stop hit — per-position SL triggered | 2026-02-23 11:20:00 | 1484.15 | 1478.99 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:45:00 | 1543.50 | 1532.02 | 0.00 | ORB-long ORB[1519.00,1534.90] vol=2.0x ATR=5.07 |
| Stop hit — per-position SL triggered | 2026-02-26 10:00:00 | 1538.43 | 1536.16 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 11:00:00 | 1473.40 | 1485.76 | 0.00 | ORB-short ORB[1486.10,1505.00] vol=5.5x ATR=4.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 11:20:00 | 1466.83 | 1484.22 | 0.00 | T1 1.5R @ 1466.83 |
| Stop hit — per-position SL triggered | 2026-03-04 13:20:00 | 1473.40 | 1477.57 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 10:40:00 | 1496.00 | 1497.61 | 0.00 | ORB-short ORB[1501.90,1510.00] vol=3.2x ATR=4.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 11:25:00 | 1489.82 | 1496.31 | 0.00 | T1 1.5R @ 1489.82 |
| Stop hit — per-position SL triggered | 2026-03-10 13:50:00 | 1496.00 | 1495.11 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 10:25:00 | 1509.10 | 1497.79 | 0.00 | ORB-long ORB[1487.50,1507.00] vol=2.6x ATR=4.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 10:50:00 | 1515.99 | 1501.84 | 0.00 | T1 1.5R @ 1515.99 |
| Target hit | 2026-03-11 15:20:00 | 1529.20 | 1528.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-18 11:15:00 | 1536.70 | 1541.41 | 0.00 | ORB-short ORB[1540.70,1562.10] vol=2.2x ATR=4.37 |
| Stop hit — per-position SL triggered | 2026-03-18 12:40:00 | 1541.07 | 1540.54 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 10:45:00 | 1530.00 | 1539.41 | 0.00 | ORB-short ORB[1534.00,1555.00] vol=3.9x ATR=6.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 12:25:00 | 1520.56 | 1535.11 | 0.00 | T1 1.5R @ 1520.56 |
| Stop hit — per-position SL triggered | 2026-03-20 15:00:00 | 1530.00 | 1529.39 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:35:00 | 1560.00 | 1548.05 | 0.00 | ORB-long ORB[1527.00,1548.80] vol=2.8x ATR=8.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 10:30:00 | 1572.11 | 1559.00 | 0.00 | T1 1.5R @ 1572.11 |
| Target hit | 2026-03-25 12:35:00 | 1583.00 | 1585.34 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — BUY (started 2026-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 1466.00 | 1459.45 | 0.00 | ORB-long ORB[1450.40,1460.00] vol=3.9x ATR=4.15 |
| Stop hit — per-position SL triggered | 2026-04-10 09:35:00 | 1461.85 | 1459.52 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 11:05:00 | 1450.50 | 1455.35 | 0.00 | ORB-short ORB[1451.80,1471.80] vol=1.9x ATR=3.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 12:30:00 | 1445.26 | 1453.41 | 0.00 | T1 1.5R @ 1445.26 |
| Stop hit — per-position SL triggered | 2026-04-15 13:00:00 | 1450.50 | 1452.91 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:45:00 | 1506.80 | 1488.66 | 0.00 | ORB-long ORB[1466.50,1479.90] vol=2.4x ATR=4.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 11:10:00 | 1513.85 | 1494.73 | 0.00 | T1 1.5R @ 1513.85 |
| Stop hit — per-position SL triggered | 2026-04-22 11:20:00 | 1506.80 | 1498.41 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:15:00 | 1536.80 | 1521.74 | 0.00 | ORB-long ORB[1486.30,1506.80] vol=1.6x ATR=8.47 |
| Stop hit — per-position SL triggered | 2026-04-23 10:35:00 | 1528.33 | 1524.03 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-24 11:10:00 | 1540.00 | 1529.48 | 0.00 | ORB-long ORB[1520.00,1536.90] vol=2.0x ATR=4.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:25:00 | 1547.16 | 1531.66 | 0.00 | T1 1.5R @ 1547.16 |
| Stop hit — per-position SL triggered | 2026-04-24 11:45:00 | 1540.00 | 1534.35 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-04-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:20:00 | 1525.60 | 1534.12 | 0.00 | ORB-short ORB[1532.50,1548.60] vol=1.6x ATR=4.58 |
| Stop hit — per-position SL triggered | 2026-04-29 10:30:00 | 1530.18 | 1533.44 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2026-04-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 09:50:00 | 1578.50 | 1565.37 | 0.00 | ORB-long ORB[1541.20,1554.70] vol=4.4x ATR=6.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 10:00:00 | 1588.88 | 1572.70 | 0.00 | T1 1.5R @ 1588.88 |
| Stop hit — per-position SL triggered | 2026-04-30 10:05:00 | 1578.50 | 1573.82 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2026-05-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:40:00 | 1521.00 | 1532.70 | 0.00 | ORB-short ORB[1524.60,1543.50] vol=2.9x ATR=5.48 |
| Stop hit — per-position SL triggered | 2026-05-05 11:10:00 | 1526.48 | 1528.82 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2026-05-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:45:00 | 1563.50 | 1551.54 | 0.00 | ORB-long ORB[1543.20,1559.60] vol=2.2x ATR=5.45 |
| Stop hit — per-position SL triggered | 2026-05-06 11:25:00 | 1558.05 | 1557.59 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2026-05-07 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:25:00 | 1587.20 | 1576.75 | 0.00 | ORB-long ORB[1572.10,1581.90] vol=1.7x ATR=6.28 |
| Stop hit — per-position SL triggered | 2026-05-07 10:40:00 | 1580.92 | 1577.25 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2026-05-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:05:00 | 1603.20 | 1589.98 | 0.00 | ORB-long ORB[1560.00,1583.80] vol=5.9x ATR=8.14 |
| Stop hit — per-position SL triggered | 2026-05-08 10:10:00 | 1595.06 | 1590.56 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 11:10:00 | 1437.30 | 2026-02-10 12:10:00 | 1431.40 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-02-10 11:10:00 | 1437.30 | 2026-02-10 15:20:00 | 1429.00 | TARGET_HIT | 0.50 | 0.58% |
| SELL | retest1 | 2026-02-18 11:00:00 | 1481.70 | 2026-02-18 12:25:00 | 1486.00 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-19 10:30:00 | 1477.40 | 2026-02-19 10:40:00 | 1481.88 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-20 10:55:00 | 1446.20 | 2026-02-20 11:20:00 | 1450.69 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-23 11:10:00 | 1488.90 | 2026-02-23 11:20:00 | 1484.15 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-26 09:45:00 | 1543.50 | 2026-02-26 10:00:00 | 1538.43 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-03-04 11:00:00 | 1473.40 | 2026-03-04 11:20:00 | 1466.83 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-03-04 11:00:00 | 1473.40 | 2026-03-04 13:20:00 | 1473.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-10 10:40:00 | 1496.00 | 2026-03-10 11:25:00 | 1489.82 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-03-10 10:40:00 | 1496.00 | 2026-03-10 13:50:00 | 1496.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-11 10:25:00 | 1509.10 | 2026-03-11 10:50:00 | 1515.99 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-03-11 10:25:00 | 1509.10 | 2026-03-11 15:20:00 | 1529.20 | TARGET_HIT | 0.50 | 1.33% |
| SELL | retest1 | 2026-03-18 11:15:00 | 1536.70 | 2026-03-18 12:40:00 | 1541.07 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-03-20 10:45:00 | 1530.00 | 2026-03-20 12:25:00 | 1520.56 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2026-03-20 10:45:00 | 1530.00 | 2026-03-20 15:00:00 | 1530.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-25 09:35:00 | 1560.00 | 2026-03-25 10:30:00 | 1572.11 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2026-03-25 09:35:00 | 1560.00 | 2026-03-25 12:35:00 | 1583.00 | TARGET_HIT | 0.50 | 1.47% |
| BUY | retest1 | 2026-04-10 09:30:00 | 1466.00 | 2026-04-10 09:35:00 | 1461.85 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-04-15 11:05:00 | 1450.50 | 2026-04-15 12:30:00 | 1445.26 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-04-15 11:05:00 | 1450.50 | 2026-04-15 13:00:00 | 1450.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 10:45:00 | 1506.80 | 2026-04-22 11:10:00 | 1513.85 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-04-22 10:45:00 | 1506.80 | 2026-04-22 11:20:00 | 1506.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-23 10:15:00 | 1536.80 | 2026-04-23 10:35:00 | 1528.33 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2026-04-24 11:10:00 | 1540.00 | 2026-04-24 11:25:00 | 1547.16 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-04-24 11:10:00 | 1540.00 | 2026-04-24 11:45:00 | 1540.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-29 10:20:00 | 1525.60 | 2026-04-29 10:30:00 | 1530.18 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-30 09:50:00 | 1578.50 | 2026-04-30 10:00:00 | 1588.88 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-04-30 09:50:00 | 1578.50 | 2026-04-30 10:05:00 | 1578.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-05 10:40:00 | 1521.00 | 2026-05-05 11:10:00 | 1526.48 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-05-06 10:45:00 | 1563.50 | 2026-05-06 11:25:00 | 1558.05 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-05-07 10:25:00 | 1587.20 | 2026-05-07 10:40:00 | 1580.92 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-05-08 10:05:00 | 1603.20 | 2026-05-08 10:10:00 | 1595.06 | STOP_HIT | 1.00 | -0.51% |
