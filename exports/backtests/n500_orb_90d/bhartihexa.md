# Bharti Hexacom Ltd. (BHARTIHEXA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1499.90
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
| ENTRY1 | 16 |
| ENTRY2 | 0 |
| PARTIAL | 11 |
| TARGET_HIT | 6 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 10
- **Target hits / Stop hits / Partials:** 6 / 10 / 11
- **Avg / median % per leg:** 0.37% / 0.38%
- **Sum % (uncompounded):** 10.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.09% | 0.9% |
| BUY @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.09% | 0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 17 | 13 | 76.5% | 5 | 4 | 8 | 0.54% | 9.1% |
| SELL @ 2nd Alert (retest1) | 17 | 13 | 76.5% | 5 | 4 | 8 | 0.54% | 9.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 27 | 17 | 63.0% | 6 | 10 | 11 | 0.37% | 10.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:50:00 | 1713.30 | 1724.85 | 0.00 | ORB-short ORB[1724.10,1735.00] vol=2.4x ATR=4.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:55:00 | 1706.73 | 1724.63 | 0.00 | T1 1.5R @ 1706.73 |
| Target hit | 2026-02-10 15:20:00 | 1678.60 | 1704.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:55:00 | 1678.20 | 1681.49 | 0.00 | ORB-short ORB[1681.00,1704.80] vol=2.5x ATR=4.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:45:00 | 1671.76 | 1679.85 | 0.00 | T1 1.5R @ 1671.76 |
| Target hit | 2026-02-24 15:20:00 | 1654.40 | 1667.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-02-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 11:00:00 | 1653.30 | 1656.74 | 0.00 | ORB-short ORB[1653.50,1664.70] vol=3.3x ATR=3.37 |
| Stop hit — per-position SL triggered | 2026-02-25 12:30:00 | 1656.67 | 1654.96 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 10:50:00 | 1586.70 | 1596.15 | 0.00 | ORB-short ORB[1590.00,1607.60] vol=1.6x ATR=4.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 11:10:00 | 1580.53 | 1593.27 | 0.00 | T1 1.5R @ 1580.53 |
| Stop hit — per-position SL triggered | 2026-03-10 11:20:00 | 1586.70 | 1592.34 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:35:00 | 1562.10 | 1573.22 | 0.00 | ORB-short ORB[1565.00,1578.90] vol=3.6x ATR=5.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:25:00 | 1554.43 | 1567.13 | 0.00 | T1 1.5R @ 1554.43 |
| Target hit | 2026-03-11 15:20:00 | 1533.40 | 1557.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2026-03-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 11:10:00 | 1526.50 | 1512.34 | 0.00 | ORB-long ORB[1501.60,1522.90] vol=2.0x ATR=4.98 |
| Stop hit — per-position SL triggered | 2026-03-12 11:15:00 | 1521.52 | 1512.49 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-19 10:20:00 | 1600.80 | 1583.51 | 0.00 | ORB-long ORB[1561.10,1584.30] vol=1.5x ATR=7.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 10:25:00 | 1612.03 | 1587.06 | 0.00 | T1 1.5R @ 1612.03 |
| Stop hit — per-position SL triggered | 2026-03-19 14:05:00 | 1600.80 | 1607.34 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:35:00 | 1595.00 | 1590.51 | 0.00 | ORB-long ORB[1580.00,1592.80] vol=2.3x ATR=6.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 14:00:00 | 1604.26 | 1595.18 | 0.00 | T1 1.5R @ 1604.26 |
| Stop hit — per-position SL triggered | 2026-03-25 14:20:00 | 1595.00 | 1595.26 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 10:15:00 | 1547.60 | 1561.82 | 0.00 | ORB-short ORB[1560.10,1582.50] vol=1.5x ATR=7.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 12:50:00 | 1535.65 | 1554.85 | 0.00 | T1 1.5R @ 1535.65 |
| Stop hit — per-position SL triggered | 2026-03-27 13:00:00 | 1547.60 | 1554.15 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:45:00 | 1580.70 | 1566.33 | 0.00 | ORB-long ORB[1549.80,1567.00] vol=1.7x ATR=6.35 |
| Stop hit — per-position SL triggered | 2026-04-08 09:50:00 | 1574.35 | 1566.93 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 10:55:00 | 1529.90 | 1538.54 | 0.00 | ORB-short ORB[1540.80,1554.80] vol=2.0x ATR=3.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 11:05:00 | 1524.61 | 1537.48 | 0.00 | T1 1.5R @ 1524.61 |
| Stop hit — per-position SL triggered | 2026-04-09 11:30:00 | 1529.90 | 1535.38 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 10:40:00 | 1529.10 | 1543.91 | 0.00 | ORB-short ORB[1530.60,1543.50] vol=2.1x ATR=4.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 15:05:00 | 1521.86 | 1532.95 | 0.00 | T1 1.5R @ 1521.86 |
| Target hit | 2026-04-10 15:20:00 | 1524.00 | 1531.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2026-04-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:55:00 | 1535.60 | 1540.10 | 0.00 | ORB-short ORB[1540.00,1560.00] vol=4.6x ATR=3.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 10:25:00 | 1529.67 | 1538.02 | 0.00 | T1 1.5R @ 1529.67 |
| Target hit | 2026-04-16 14:25:00 | 1534.50 | 1533.76 | 0.00 | Trail-exit close>VWAP |

### Cycle 14 — BUY (started 2026-04-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-20 10:50:00 | 1570.70 | 1547.94 | 0.00 | ORB-long ORB[1527.60,1547.90] vol=5.8x ATR=6.62 |
| Stop hit — per-position SL triggered | 2026-04-20 10:55:00 | 1564.08 | 1552.03 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 1554.30 | 1544.75 | 0.00 | ORB-long ORB[1530.00,1551.20] vol=1.7x ATR=5.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:50:00 | 1562.03 | 1553.48 | 0.00 | T1 1.5R @ 1562.03 |
| Target hit | 2026-04-21 13:30:00 | 1563.00 | 1563.22 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — BUY (started 2026-04-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:00:00 | 1569.90 | 1558.13 | 0.00 | ORB-long ORB[1546.10,1563.00] vol=4.3x ATR=4.64 |
| Stop hit — per-position SL triggered | 2026-04-23 10:20:00 | 1565.26 | 1562.54 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 10:50:00 | 1713.30 | 2026-02-10 10:55:00 | 1706.73 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-02-10 10:50:00 | 1713.30 | 2026-02-10 15:20:00 | 1678.60 | TARGET_HIT | 0.50 | 2.03% |
| SELL | retest1 | 2026-02-24 10:55:00 | 1678.20 | 2026-02-24 11:45:00 | 1671.76 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-02-24 10:55:00 | 1678.20 | 2026-02-24 15:20:00 | 1654.40 | TARGET_HIT | 0.50 | 1.42% |
| SELL | retest1 | 2026-02-25 11:00:00 | 1653.30 | 2026-02-25 12:30:00 | 1656.67 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-03-10 10:50:00 | 1586.70 | 2026-03-10 11:10:00 | 1580.53 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-03-10 10:50:00 | 1586.70 | 2026-03-10 11:20:00 | 1586.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-11 10:35:00 | 1562.10 | 2026-03-11 11:25:00 | 1554.43 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-03-11 10:35:00 | 1562.10 | 2026-03-11 15:20:00 | 1533.40 | TARGET_HIT | 0.50 | 1.84% |
| BUY | retest1 | 2026-03-12 11:10:00 | 1526.50 | 2026-03-12 11:15:00 | 1521.52 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-03-19 10:20:00 | 1600.80 | 2026-03-19 10:25:00 | 1612.03 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-03-19 10:20:00 | 1600.80 | 2026-03-19 14:05:00 | 1600.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-25 09:35:00 | 1595.00 | 2026-03-25 14:00:00 | 1604.26 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-03-25 09:35:00 | 1595.00 | 2026-03-25 14:20:00 | 1595.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-27 10:15:00 | 1547.60 | 2026-03-27 12:50:00 | 1535.65 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2026-03-27 10:15:00 | 1547.60 | 2026-03-27 13:00:00 | 1547.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-08 09:45:00 | 1580.70 | 2026-04-08 09:50:00 | 1574.35 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-04-09 10:55:00 | 1529.90 | 2026-04-09 11:05:00 | 1524.61 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-04-09 10:55:00 | 1529.90 | 2026-04-09 11:30:00 | 1529.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-10 10:40:00 | 1529.10 | 2026-04-10 15:05:00 | 1521.86 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-04-10 10:40:00 | 1529.10 | 2026-04-10 15:20:00 | 1524.00 | TARGET_HIT | 0.50 | 0.33% |
| SELL | retest1 | 2026-04-16 09:55:00 | 1535.60 | 2026-04-16 10:25:00 | 1529.67 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-04-16 09:55:00 | 1535.60 | 2026-04-16 14:25:00 | 1534.50 | TARGET_HIT | 0.50 | 0.07% |
| BUY | retest1 | 2026-04-20 10:50:00 | 1570.70 | 2026-04-20 10:55:00 | 1564.08 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-04-21 09:35:00 | 1554.30 | 2026-04-21 09:50:00 | 1562.03 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-04-21 09:35:00 | 1554.30 | 2026-04-21 13:30:00 | 1563.00 | TARGET_HIT | 0.50 | 0.56% |
| BUY | retest1 | 2026-04-23 10:00:00 | 1569.90 | 2026-04-23 10:20:00 | 1565.26 | STOP_HIT | 1.00 | -0.30% |
