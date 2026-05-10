# LG Electronics India Ltd. (LGEINDIA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1508.60
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
| ENTRY1 | 19 |
| ENTRY2 | 0 |
| PARTIAL | 10 |
| TARGET_HIT | 5 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 14
- **Target hits / Stop hits / Partials:** 5 / 14 / 10
- **Avg / median % per leg:** 0.15% / 0.26%
- **Sum % (uncompounded):** 4.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 7 | 41.2% | 2 | 10 | 5 | 0.03% | 0.6% |
| BUY @ 2nd Alert (retest1) | 17 | 7 | 41.2% | 2 | 10 | 5 | 0.03% | 0.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 8 | 66.7% | 3 | 4 | 5 | 0.32% | 3.8% |
| SELL @ 2nd Alert (retest1) | 12 | 8 | 66.7% | 3 | 4 | 5 | 0.32% | 3.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 29 | 15 | 51.7% | 5 | 14 | 10 | 0.15% | 4.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:30:00 | 1553.70 | 1561.80 | 0.00 | ORB-short ORB[1556.80,1576.00] vol=2.4x ATR=5.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:50:00 | 1545.92 | 1558.48 | 0.00 | T1 1.5R @ 1545.92 |
| Stop hit — per-position SL triggered | 2026-02-18 11:45:00 | 1553.70 | 1553.64 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:55:00 | 1574.50 | 1566.27 | 0.00 | ORB-long ORB[1550.40,1563.40] vol=1.6x ATR=5.25 |
| Stop hit — per-position SL triggered | 2026-02-19 10:10:00 | 1569.25 | 1566.72 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:25:00 | 1554.90 | 1544.97 | 0.00 | ORB-long ORB[1530.10,1545.00] vol=1.6x ATR=5.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 10:35:00 | 1562.44 | 1547.12 | 0.00 | T1 1.5R @ 1562.44 |
| Stop hit — per-position SL triggered | 2026-02-20 11:55:00 | 1554.90 | 1550.76 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 10:20:00 | 1565.50 | 1550.69 | 0.00 | ORB-long ORB[1541.80,1554.90] vol=1.5x ATR=4.97 |
| Stop hit — per-position SL triggered | 2026-02-23 10:25:00 | 1560.53 | 1552.13 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:55:00 | 1570.60 | 1563.56 | 0.00 | ORB-long ORB[1553.00,1565.90] vol=3.7x ATR=4.93 |
| Stop hit — per-position SL triggered | 2026-02-24 10:05:00 | 1565.67 | 1564.11 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 09:35:00 | 1580.20 | 1568.03 | 0.00 | ORB-long ORB[1551.20,1564.50] vol=4.6x ATR=5.80 |
| Stop hit — per-position SL triggered | 2026-02-27 12:10:00 | 1574.40 | 1573.63 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 11:15:00 | 1575.00 | 1567.41 | 0.00 | ORB-long ORB[1556.70,1574.90] vol=7.4x ATR=3.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:20:00 | 1580.83 | 1568.30 | 0.00 | T1 1.5R @ 1580.83 |
| Stop hit — per-position SL triggered | 2026-03-05 11:25:00 | 1575.00 | 1569.11 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:40:00 | 1562.80 | 1554.01 | 0.00 | ORB-long ORB[1548.80,1560.50] vol=1.9x ATR=4.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 10:50:00 | 1570.11 | 1557.63 | 0.00 | T1 1.5R @ 1570.11 |
| Target hit | 2026-03-10 15:10:00 | 1571.50 | 1572.97 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2026-03-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:35:00 | 1586.50 | 1572.66 | 0.00 | ORB-long ORB[1560.20,1572.90] vol=3.6x ATR=5.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 09:45:00 | 1594.71 | 1577.56 | 0.00 | T1 1.5R @ 1594.71 |
| Target hit | 2026-03-11 10:05:00 | 1592.50 | 1592.79 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2026-03-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 09:30:00 | 1577.50 | 1569.59 | 0.00 | ORB-long ORB[1559.30,1575.00] vol=2.1x ATR=5.40 |
| Stop hit — per-position SL triggered | 2026-03-13 09:55:00 | 1572.10 | 1571.83 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 09:45:00 | 1549.10 | 1556.95 | 0.00 | ORB-short ORB[1554.90,1568.90] vol=4.0x ATR=5.00 |
| Stop hit — per-position SL triggered | 2026-03-19 10:35:00 | 1554.10 | 1553.86 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 10:10:00 | 1552.00 | 1544.70 | 0.00 | ORB-long ORB[1534.30,1546.60] vol=1.8x ATR=4.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 10:55:00 | 1559.26 | 1547.63 | 0.00 | T1 1.5R @ 1559.26 |
| Stop hit — per-position SL triggered | 2026-03-20 12:55:00 | 1552.00 | 1554.12 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-22 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:05:00 | 1650.00 | 1633.49 | 0.00 | ORB-long ORB[1607.80,1620.40] vol=11.5x ATR=8.84 |
| Stop hit — per-position SL triggered | 2026-04-22 10:10:00 | 1641.16 | 1633.92 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 10:55:00 | 1559.10 | 1581.31 | 0.00 | ORB-short ORB[1587.70,1608.10] vol=2.5x ATR=4.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 11:20:00 | 1551.98 | 1573.79 | 0.00 | T1 1.5R @ 1551.98 |
| Target hit | 2026-04-23 15:20:00 | 1547.00 | 1556.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2026-04-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:20:00 | 1630.40 | 1609.72 | 0.00 | ORB-long ORB[1586.10,1607.00] vol=7.0x ATR=7.49 |
| Stop hit — per-position SL triggered | 2026-04-27 10:25:00 | 1622.91 | 1615.96 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-05-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 11:05:00 | 1583.00 | 1597.03 | 0.00 | ORB-short ORB[1595.00,1612.50] vol=6.1x ATR=4.72 |
| Stop hit — per-position SL triggered | 2026-05-04 11:50:00 | 1587.72 | 1592.91 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-05-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:50:00 | 1574.00 | 1580.56 | 0.00 | ORB-short ORB[1575.10,1589.20] vol=1.8x ATR=3.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:00:00 | 1568.44 | 1564.73 | 0.00 | T1 1.5R @ 1568.44 |
| Target hit | 2026-05-05 13:05:00 | 1564.60 | 1563.52 | 0.00 | Trail-exit close>VWAP |

### Cycle 18 — SELL (started 2026-05-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 11:10:00 | 1557.70 | 1560.72 | 0.00 | ORB-short ORB[1562.20,1576.30] vol=8.7x ATR=4.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 11:35:00 | 1551.23 | 1559.75 | 0.00 | T1 1.5R @ 1551.23 |
| Target hit | 2026-05-07 15:20:00 | 1541.30 | 1547.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 11:15:00 | 1524.30 | 1532.38 | 0.00 | ORB-short ORB[1542.60,1553.60] vol=1.5x ATR=2.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 11:25:00 | 1520.26 | 1530.52 | 0.00 | T1 1.5R @ 1520.26 |
| Stop hit — per-position SL triggered | 2026-05-08 12:40:00 | 1524.30 | 1523.20 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-18 09:30:00 | 1553.70 | 2026-02-18 09:50:00 | 1545.92 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-02-18 09:30:00 | 1553.70 | 2026-02-18 11:45:00 | 1553.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-19 09:55:00 | 1574.50 | 2026-02-19 10:10:00 | 1569.25 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-20 10:25:00 | 1554.90 | 2026-02-20 10:35:00 | 1562.44 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-02-20 10:25:00 | 1554.90 | 2026-02-20 11:55:00 | 1554.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-23 10:20:00 | 1565.50 | 2026-02-23 10:25:00 | 1560.53 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-24 09:55:00 | 1570.60 | 2026-02-24 10:05:00 | 1565.67 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-27 09:35:00 | 1580.20 | 2026-02-27 12:10:00 | 1574.40 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-03-05 11:15:00 | 1575.00 | 2026-03-05 11:20:00 | 1580.83 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-03-05 11:15:00 | 1575.00 | 2026-03-05 11:25:00 | 1575.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-10 10:40:00 | 1562.80 | 2026-03-10 10:50:00 | 1570.11 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-03-10 10:40:00 | 1562.80 | 2026-03-10 15:10:00 | 1571.50 | TARGET_HIT | 0.50 | 0.56% |
| BUY | retest1 | 2026-03-11 09:35:00 | 1586.50 | 2026-03-11 09:45:00 | 1594.71 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-03-11 09:35:00 | 1586.50 | 2026-03-11 10:05:00 | 1592.50 | TARGET_HIT | 0.50 | 0.38% |
| BUY | retest1 | 2026-03-13 09:30:00 | 1577.50 | 2026-03-13 09:55:00 | 1572.10 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-19 09:45:00 | 1549.10 | 2026-03-19 10:35:00 | 1554.10 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-03-20 10:10:00 | 1552.00 | 2026-03-20 10:55:00 | 1559.26 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-03-20 10:10:00 | 1552.00 | 2026-03-20 12:55:00 | 1552.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 10:05:00 | 1650.00 | 2026-04-22 10:10:00 | 1641.16 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2026-04-23 10:55:00 | 1559.10 | 2026-04-23 11:20:00 | 1551.98 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-04-23 10:55:00 | 1559.10 | 2026-04-23 15:20:00 | 1547.00 | TARGET_HIT | 0.50 | 0.78% |
| BUY | retest1 | 2026-04-27 10:20:00 | 1630.40 | 2026-04-27 10:25:00 | 1622.91 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-05-04 11:05:00 | 1583.00 | 2026-05-04 11:50:00 | 1587.72 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-05-05 10:50:00 | 1574.00 | 2026-05-05 11:00:00 | 1568.44 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-05-05 10:50:00 | 1574.00 | 2026-05-05 13:05:00 | 1564.60 | TARGET_HIT | 0.50 | 0.60% |
| SELL | retest1 | 2026-05-07 11:10:00 | 1557.70 | 2026-05-07 11:35:00 | 1551.23 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-05-07 11:10:00 | 1557.70 | 2026-05-07 15:20:00 | 1541.30 | TARGET_HIT | 0.50 | 1.05% |
| SELL | retest1 | 2026-05-08 11:15:00 | 1524.30 | 2026-05-08 11:25:00 | 1520.26 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2026-05-08 11:15:00 | 1524.30 | 2026-05-08 12:40:00 | 1524.30 | STOP_HIT | 0.50 | 0.00% |
