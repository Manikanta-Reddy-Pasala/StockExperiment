# Bombay Burmah Trading Corporation Ltd. (BBTC)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1563.30
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
| PARTIAL | 6 |
| TARGET_HIT | 1 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 14
- **Target hits / Stop hits / Partials:** 1 / 14 / 6
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 2.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 3 | 30.0% | 0 | 7 | 3 | 0.05% | 0.5% |
| BUY @ 2nd Alert (retest1) | 10 | 3 | 30.0% | 0 | 7 | 3 | 0.05% | 0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.16% | 1.8% |
| SELL @ 2nd Alert (retest1) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.16% | 1.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 21 | 7 | 33.3% | 1 | 14 | 6 | 0.11% | 2.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:55:00 | 1823.50 | 1813.14 | 0.00 | ORB-long ORB[1796.00,1820.10] vol=2.2x ATR=5.54 |
| Stop hit — per-position SL triggered | 2026-02-17 10:50:00 | 1817.96 | 1815.75 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:15:00 | 1805.20 | 1815.09 | 0.00 | ORB-short ORB[1811.60,1830.50] vol=3.8x ATR=3.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 12:15:00 | 1800.18 | 1812.86 | 0.00 | T1 1.5R @ 1800.18 |
| Stop hit — per-position SL triggered | 2026-02-19 12:25:00 | 1805.20 | 1812.69 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 11:15:00 | 1607.00 | 1617.66 | 0.00 | ORB-short ORB[1609.80,1633.80] vol=2.2x ATR=4.54 |
| Stop hit — per-position SL triggered | 2026-03-06 11:40:00 | 1611.54 | 1616.99 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 10:05:00 | 1532.90 | 1546.03 | 0.00 | ORB-short ORB[1549.50,1568.40] vol=1.5x ATR=6.87 |
| Stop hit — per-position SL triggered | 2026-03-10 10:20:00 | 1539.77 | 1542.50 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 11:05:00 | 1582.10 | 1572.31 | 0.00 | ORB-long ORB[1564.10,1577.00] vol=4.3x ATR=4.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:15:00 | 1589.00 | 1574.56 | 0.00 | T1 1.5R @ 1589.00 |
| Stop hit — per-position SL triggered | 2026-03-11 11:20:00 | 1582.10 | 1574.65 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:15:00 | 1498.50 | 1512.27 | 0.00 | ORB-short ORB[1505.70,1527.30] vol=4.7x ATR=6.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:30:00 | 1488.99 | 1508.77 | 0.00 | T1 1.5R @ 1488.99 |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 1498.50 | 1502.89 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:45:00 | 1540.00 | 1531.77 | 0.00 | ORB-long ORB[1515.00,1537.50] vol=3.9x ATR=5.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 09:50:00 | 1548.84 | 1533.62 | 0.00 | T1 1.5R @ 1548.84 |
| Stop hit — per-position SL triggered | 2026-03-18 09:55:00 | 1540.00 | 1534.01 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:40:00 | 1448.70 | 1435.90 | 0.00 | ORB-long ORB[1421.00,1438.00] vol=1.7x ATR=5.27 |
| Stop hit — per-position SL triggered | 2026-03-25 10:55:00 | 1443.43 | 1436.41 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-06 09:55:00 | 1418.50 | 1409.09 | 0.00 | ORB-long ORB[1400.00,1417.00] vol=1.7x ATR=8.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-06 10:45:00 | 1430.58 | 1413.93 | 0.00 | T1 1.5R @ 1430.58 |
| Stop hit — per-position SL triggered | 2026-04-06 11:10:00 | 1418.50 | 1415.18 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:55:00 | 1449.10 | 1466.03 | 0.00 | ORB-short ORB[1464.10,1484.00] vol=1.9x ATR=7.29 |
| Stop hit — per-position SL triggered | 2026-04-09 10:30:00 | 1456.39 | 1463.06 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:55:00 | 1479.80 | 1488.59 | 0.00 | ORB-short ORB[1490.00,1510.00] vol=3.3x ATR=4.63 |
| Stop hit — per-position SL triggered | 2026-04-16 10:30:00 | 1484.43 | 1486.10 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:45:00 | 1557.00 | 1546.37 | 0.00 | ORB-long ORB[1535.40,1552.00] vol=2.4x ATR=6.01 |
| Stop hit — per-position SL triggered | 2026-04-23 10:10:00 | 1550.99 | 1547.17 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 1523.80 | 1533.64 | 0.00 | ORB-short ORB[1531.50,1549.70] vol=1.9x ATR=5.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:40:00 | 1515.68 | 1528.99 | 0.00 | T1 1.5R @ 1515.68 |
| Target hit | 2026-04-24 15:20:00 | 1490.30 | 1508.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2026-05-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:55:00 | 1511.60 | 1521.25 | 0.00 | ORB-short ORB[1516.30,1533.90] vol=2.5x ATR=4.30 |
| Stop hit — per-position SL triggered | 2026-05-05 11:05:00 | 1515.90 | 1520.92 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:30:00 | 1535.00 | 1527.87 | 0.00 | ORB-long ORB[1520.00,1531.70] vol=1.9x ATR=5.09 |
| Stop hit — per-position SL triggered | 2026-05-06 09:35:00 | 1529.91 | 1528.46 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-17 09:55:00 | 1823.50 | 2026-02-17 10:50:00 | 1817.96 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-19 11:15:00 | 1805.20 | 2026-02-19 12:15:00 | 1800.18 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2026-02-19 11:15:00 | 1805.20 | 2026-02-19 12:25:00 | 1805.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-06 11:15:00 | 1607.00 | 2026-03-06 11:40:00 | 1611.54 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-03-10 10:05:00 | 1532.90 | 2026-03-10 10:20:00 | 1539.77 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-03-11 11:05:00 | 1582.10 | 2026-03-11 11:15:00 | 1589.00 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-03-11 11:05:00 | 1582.10 | 2026-03-11 11:20:00 | 1582.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-16 10:15:00 | 1498.50 | 2026-03-16 10:30:00 | 1488.99 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2026-03-16 10:15:00 | 1498.50 | 2026-03-16 11:15:00 | 1498.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-18 09:45:00 | 1540.00 | 2026-03-18 09:50:00 | 1548.84 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-03-18 09:45:00 | 1540.00 | 2026-03-18 09:55:00 | 1540.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-25 10:40:00 | 1448.70 | 2026-03-25 10:55:00 | 1443.43 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-06 09:55:00 | 1418.50 | 2026-04-06 10:45:00 | 1430.58 | PARTIAL | 0.50 | 0.85% |
| BUY | retest1 | 2026-04-06 09:55:00 | 1418.50 | 2026-04-06 11:10:00 | 1418.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-09 09:55:00 | 1449.10 | 2026-04-09 10:30:00 | 1456.39 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2026-04-16 09:55:00 | 1479.80 | 2026-04-16 10:30:00 | 1484.43 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-23 09:45:00 | 1557.00 | 2026-04-23 10:10:00 | 1550.99 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-04-24 09:30:00 | 1523.80 | 2026-04-24 09:40:00 | 1515.68 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-04-24 09:30:00 | 1523.80 | 2026-04-24 15:20:00 | 1490.30 | TARGET_HIT | 0.50 | 2.20% |
| SELL | retest1 | 2026-05-05 10:55:00 | 1511.60 | 2026-05-05 11:05:00 | 1515.90 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-05-06 09:30:00 | 1535.00 | 2026-05-06 09:35:00 | 1529.91 | STOP_HIT | 1.00 | -0.33% |
