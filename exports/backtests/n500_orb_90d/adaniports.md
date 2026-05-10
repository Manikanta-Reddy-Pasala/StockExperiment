# Adani Ports and Special Economic Zone Ltd. (ADANIPORTS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1760.00
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
| ENTRY1 | 14 |
| ENTRY2 | 0 |
| PARTIAL | 10 |
| TARGET_HIT | 3 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 11
- **Target hits / Stop hits / Partials:** 3 / 11 / 10
- **Avg / median % per leg:** 0.18% / 0.15%
- **Sum % (uncompounded):** 4.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 7 | 53.8% | 1 | 6 | 6 | 0.17% | 2.1% |
| BUY @ 2nd Alert (retest1) | 13 | 7 | 53.8% | 1 | 6 | 6 | 0.17% | 2.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 6 | 54.5% | 2 | 5 | 4 | 0.19% | 2.1% |
| SELL @ 2nd Alert (retest1) | 11 | 6 | 54.5% | 2 | 5 | 4 | 0.19% | 2.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 24 | 13 | 54.2% | 3 | 11 | 10 | 0.18% | 4.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:35:00 | 1543.80 | 1555.20 | 0.00 | ORB-short ORB[1557.00,1567.00] vol=3.3x ATR=4.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:40:00 | 1536.69 | 1549.69 | 0.00 | T1 1.5R @ 1536.69 |
| Stop hit — per-position SL triggered | 2026-02-10 10:50:00 | 1543.80 | 1548.51 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 10:40:00 | 1549.90 | 1550.41 | 0.00 | ORB-short ORB[1550.40,1559.40] vol=2.5x ATR=3.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 10:45:00 | 1545.28 | 1550.09 | 0.00 | T1 1.5R @ 1545.28 |
| Target hit | 2026-02-11 12:50:00 | 1547.60 | 1546.26 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — SELL (started 2026-02-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:10:00 | 1539.30 | 1544.82 | 0.00 | ORB-short ORB[1541.90,1551.00] vol=3.6x ATR=2.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:20:00 | 1535.68 | 1544.16 | 0.00 | T1 1.5R @ 1535.68 |
| Stop hit — per-position SL triggered | 2026-02-12 12:50:00 | 1539.30 | 1541.27 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:45:00 | 1514.70 | 1525.44 | 0.00 | ORB-short ORB[1525.70,1543.00] vol=2.6x ATR=4.42 |
| Stop hit — per-position SL triggered | 2026-02-13 12:00:00 | 1519.12 | 1519.22 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:10:00 | 1557.30 | 1545.44 | 0.00 | ORB-long ORB[1535.70,1545.00] vol=1.9x ATR=4.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:45:00 | 1563.45 | 1556.74 | 0.00 | T1 1.5R @ 1563.45 |
| Stop hit — per-position SL triggered | 2026-02-17 10:55:00 | 1557.30 | 1556.93 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:00:00 | 1574.30 | 1567.61 | 0.00 | ORB-long ORB[1563.90,1571.80] vol=1.9x ATR=3.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:05:00 | 1579.98 | 1570.07 | 0.00 | T1 1.5R @ 1579.98 |
| Stop hit — per-position SL triggered | 2026-02-25 10:20:00 | 1574.30 | 1573.01 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:00:00 | 1530.10 | 1535.92 | 0.00 | ORB-short ORB[1534.10,1545.10] vol=1.6x ATR=3.53 |
| Stop hit — per-position SL triggered | 2026-02-27 10:25:00 | 1533.63 | 1535.06 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 11:10:00 | 1356.40 | 1368.68 | 0.00 | ORB-short ORB[1370.00,1390.00] vol=2.0x ATR=4.60 |
| Stop hit — per-position SL triggered | 2026-03-13 11:25:00 | 1361.00 | 1368.13 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 10:45:00 | 1378.70 | 1374.50 | 0.00 | ORB-long ORB[1362.40,1378.50] vol=1.7x ATR=3.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 11:15:00 | 1384.52 | 1375.31 | 0.00 | T1 1.5R @ 1384.52 |
| Stop hit — per-position SL triggered | 2026-03-20 11:55:00 | 1378.70 | 1377.10 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 09:30:00 | 1355.50 | 1361.24 | 0.00 | ORB-short ORB[1356.30,1375.00] vol=1.8x ATR=5.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 10:10:00 | 1348.00 | 1357.13 | 0.00 | T1 1.5R @ 1348.00 |
| Target hit | 2026-03-27 15:20:00 | 1337.90 | 1343.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2026-04-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:45:00 | 1533.40 | 1523.04 | 0.00 | ORB-long ORB[1510.00,1524.90] vol=1.6x ATR=5.10 |
| Stop hit — per-position SL triggered | 2026-04-16 09:55:00 | 1528.30 | 1524.37 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:55:00 | 1585.60 | 1578.71 | 0.00 | ORB-long ORB[1566.20,1581.20] vol=3.4x ATR=4.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 11:00:00 | 1591.72 | 1581.25 | 0.00 | T1 1.5R @ 1591.72 |
| Stop hit — per-position SL triggered | 2026-04-23 11:10:00 | 1585.60 | 1581.69 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:40:00 | 1664.00 | 1652.36 | 0.00 | ORB-long ORB[1644.90,1659.00] vol=3.8x ATR=4.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:50:00 | 1670.83 | 1655.75 | 0.00 | T1 1.5R @ 1670.83 |
| Stop hit — per-position SL triggered | 2026-04-29 11:15:00 | 1664.00 | 1658.12 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:30:00 | 1760.40 | 1750.35 | 0.00 | ORB-long ORB[1727.70,1753.10] vol=5.9x ATR=5.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 10:05:00 | 1768.72 | 1754.68 | 0.00 | T1 1.5R @ 1768.72 |
| Target hit | 2026-05-08 11:45:00 | 1761.00 | 1762.00 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 10:35:00 | 1543.80 | 2026-02-10 10:40:00 | 1536.69 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-02-10 10:35:00 | 1543.80 | 2026-02-10 10:50:00 | 1543.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-11 10:40:00 | 1549.90 | 2026-02-11 10:45:00 | 1545.28 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2026-02-11 10:40:00 | 1549.90 | 2026-02-11 12:50:00 | 1547.60 | TARGET_HIT | 0.50 | 0.15% |
| SELL | retest1 | 2026-02-12 11:10:00 | 1539.30 | 2026-02-12 11:20:00 | 1535.68 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2026-02-12 11:10:00 | 1539.30 | 2026-02-12 12:50:00 | 1539.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-13 09:45:00 | 1514.70 | 2026-02-13 12:00:00 | 1519.12 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-17 10:10:00 | 1557.30 | 2026-02-17 10:45:00 | 1563.45 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-17 10:10:00 | 1557.30 | 2026-02-17 10:55:00 | 1557.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 10:00:00 | 1574.30 | 2026-02-25 10:05:00 | 1579.98 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-02-25 10:00:00 | 1574.30 | 2026-02-25 10:20:00 | 1574.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 10:00:00 | 1530.10 | 2026-02-27 10:25:00 | 1533.63 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-03-13 11:10:00 | 1356.40 | 2026-03-13 11:25:00 | 1361.00 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-03-20 10:45:00 | 1378.70 | 2026-03-20 11:15:00 | 1384.52 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-03-20 10:45:00 | 1378.70 | 2026-03-20 11:55:00 | 1378.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-27 09:30:00 | 1355.50 | 2026-03-27 10:10:00 | 1348.00 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-03-27 09:30:00 | 1355.50 | 2026-03-27 15:20:00 | 1337.90 | TARGET_HIT | 0.50 | 1.30% |
| BUY | retest1 | 2026-04-16 09:45:00 | 1533.40 | 2026-04-16 09:55:00 | 1528.30 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-23 10:55:00 | 1585.60 | 2026-04-23 11:00:00 | 1591.72 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-04-23 10:55:00 | 1585.60 | 2026-04-23 11:10:00 | 1585.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 10:40:00 | 1664.00 | 2026-04-29 10:50:00 | 1670.83 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-04-29 10:40:00 | 1664.00 | 2026-04-29 11:15:00 | 1664.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-08 09:30:00 | 1760.40 | 2026-05-08 10:05:00 | 1768.72 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-05-08 09:30:00 | 1760.40 | 2026-05-08 11:45:00 | 1761.00 | TARGET_HIT | 0.50 | 0.03% |
