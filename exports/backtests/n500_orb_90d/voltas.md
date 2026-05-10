# Voltas Ltd. (VOLTAS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1323.00
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
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 12
- **Target hits / Stop hits / Partials:** 2 / 12 / 4
- **Avg / median % per leg:** 0.02% / -0.23%
- **Sum % (uncompounded):** 0.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 3 | 30.0% | 1 | 7 | 2 | -0.04% | -0.4% |
| BUY @ 2nd Alert (retest1) | 10 | 3 | 30.0% | 1 | 7 | 2 | -0.04% | -0.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 3 | 37.5% | 1 | 5 | 2 | 0.09% | 0.7% |
| SELL @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 1 | 5 | 2 | 0.09% | 0.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 18 | 6 | 33.3% | 2 | 12 | 4 | 0.02% | 0.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:05:00 | 1493.70 | 1481.47 | 0.00 | ORB-long ORB[1477.40,1486.10] vol=2.1x ATR=5.20 |
| Stop hit — per-position SL triggered | 2026-02-11 10:20:00 | 1488.50 | 1483.47 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:10:00 | 1515.00 | 1500.60 | 0.00 | ORB-long ORB[1492.00,1502.30] vol=1.6x ATR=5.29 |
| Stop hit — per-position SL triggered | 2026-02-12 10:25:00 | 1509.71 | 1502.22 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:50:00 | 1516.60 | 1520.08 | 0.00 | ORB-short ORB[1516.70,1534.00] vol=1.8x ATR=5.26 |
| Stop hit — per-position SL triggered | 2026-02-13 11:25:00 | 1521.86 | 1519.59 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 10:45:00 | 1519.90 | 1528.32 | 0.00 | ORB-short ORB[1521.80,1540.00] vol=3.7x ATR=4.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 12:05:00 | 1513.24 | 1526.01 | 0.00 | T1 1.5R @ 1513.24 |
| Stop hit — per-position SL triggered | 2026-02-17 12:35:00 | 1519.90 | 1525.23 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:15:00 | 1523.60 | 1537.64 | 0.00 | ORB-short ORB[1534.90,1552.80] vol=4.4x ATR=4.22 |
| Stop hit — per-position SL triggered | 2026-02-18 11:55:00 | 1527.82 | 1533.78 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:45:00 | 1549.20 | 1542.72 | 0.00 | ORB-long ORB[1534.10,1549.10] vol=1.6x ATR=5.33 |
| Stop hit — per-position SL triggered | 2026-02-24 12:30:00 | 1543.87 | 1546.98 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:30:00 | 1558.00 | 1549.76 | 0.00 | ORB-long ORB[1535.50,1549.90] vol=3.4x ATR=4.92 |
| Stop hit — per-position SL triggered | 2026-02-25 10:55:00 | 1553.08 | 1551.07 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-02-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:35:00 | 1536.10 | 1531.90 | 0.00 | ORB-long ORB[1521.00,1533.00] vol=1.8x ATR=4.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 09:55:00 | 1543.26 | 1534.34 | 0.00 | T1 1.5R @ 1543.26 |
| Target hit | 2026-02-26 11:40:00 | 1544.50 | 1544.78 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2026-02-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 11:10:00 | 1568.00 | 1551.62 | 0.00 | ORB-long ORB[1534.20,1555.50] vol=3.1x ATR=4.64 |
| Stop hit — per-position SL triggered | 2026-02-27 11:35:00 | 1563.36 | 1554.42 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:55:00 | 1467.70 | 1480.47 | 0.00 | ORB-short ORB[1472.60,1487.40] vol=1.9x ATR=4.64 |
| Stop hit — per-position SL triggered | 2026-03-06 11:20:00 | 1472.34 | 1477.20 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:35:00 | 1421.40 | 1432.67 | 0.00 | ORB-short ORB[1430.00,1448.90] vol=2.9x ATR=6.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:40:00 | 1412.00 | 1421.13 | 0.00 | T1 1.5R @ 1412.00 |
| Target hit | 2026-03-13 11:25:00 | 1409.00 | 1404.00 | 0.00 | Trail-exit close>VWAP |

### Cycle 12 — BUY (started 2026-04-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:55:00 | 1465.00 | 1458.15 | 0.00 | ORB-long ORB[1444.40,1456.00] vol=1.7x ATR=3.34 |
| Stop hit — per-position SL triggered | 2026-04-21 12:40:00 | 1461.66 | 1462.14 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-22 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:20:00 | 1482.50 | 1472.32 | 0.00 | ORB-long ORB[1455.00,1476.20] vol=1.9x ATR=4.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 10:25:00 | 1489.35 | 1474.23 | 0.00 | T1 1.5R @ 1489.35 |
| Stop hit — per-position SL triggered | 2026-04-22 15:05:00 | 1482.50 | 1482.33 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:55:00 | 1436.10 | 1443.21 | 0.00 | ORB-short ORB[1443.30,1457.90] vol=1.8x ATR=4.22 |
| Stop hit — per-position SL triggered | 2026-04-24 11:15:00 | 1440.32 | 1442.72 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-11 10:05:00 | 1493.70 | 2026-02-11 10:20:00 | 1488.50 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-12 10:10:00 | 1515.00 | 2026-02-12 10:25:00 | 1509.71 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-13 10:50:00 | 1516.60 | 2026-02-13 11:25:00 | 1521.86 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-17 10:45:00 | 1519.90 | 2026-02-17 12:05:00 | 1513.24 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-02-17 10:45:00 | 1519.90 | 2026-02-17 12:35:00 | 1519.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-18 11:15:00 | 1523.60 | 2026-02-18 11:55:00 | 1527.82 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-24 09:45:00 | 1549.20 | 2026-02-24 12:30:00 | 1543.87 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-25 10:30:00 | 1558.00 | 2026-02-25 10:55:00 | 1553.08 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-26 09:35:00 | 1536.10 | 2026-02-26 09:55:00 | 1543.26 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-02-26 09:35:00 | 1536.10 | 2026-02-26 11:40:00 | 1544.50 | TARGET_HIT | 0.50 | 0.55% |
| BUY | retest1 | 2026-02-27 11:10:00 | 1568.00 | 2026-02-27 11:35:00 | 1563.36 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-06 10:55:00 | 1467.70 | 2026-03-06 11:20:00 | 1472.34 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-03-13 09:35:00 | 1421.40 | 2026-03-13 09:40:00 | 1412.00 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2026-03-13 09:35:00 | 1421.40 | 2026-03-13 11:25:00 | 1409.00 | TARGET_HIT | 0.50 | 0.87% |
| BUY | retest1 | 2026-04-21 10:55:00 | 1465.00 | 2026-04-21 12:40:00 | 1461.66 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-04-22 10:20:00 | 1482.50 | 2026-04-22 10:25:00 | 1489.35 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-04-22 10:20:00 | 1482.50 | 2026-04-22 15:05:00 | 1482.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-24 10:55:00 | 1436.10 | 2026-04-24 11:15:00 | 1440.32 | STOP_HIT | 1.00 | -0.29% |
