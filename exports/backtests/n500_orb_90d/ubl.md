# United Breweries Ltd. (UBL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1419.00
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
| TARGET_HIT | 3 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 11
- **Target hits / Stop hits / Partials:** 3 / 11 / 4
- **Avg / median % per leg:** 0.18% / -0.18%
- **Sum % (uncompounded):** 3.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 7 | 50.0% | 3 | 7 | 4 | 0.30% | 4.2% |
| BUY @ 2nd Alert (retest1) | 14 | 7 | 50.0% | 3 | 7 | 4 | 0.30% | 4.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.23% | -0.9% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.23% | -0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 18 | 7 | 38.9% | 3 | 11 | 4 | 0.18% | 3.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:15:00 | 1610.10 | 1604.43 | 0.00 | ORB-long ORB[1594.00,1604.50] vol=1.5x ATR=3.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:25:00 | 1615.27 | 1608.20 | 0.00 | T1 1.5R @ 1615.27 |
| Target hit | 2026-02-18 11:30:00 | 1612.70 | 1612.74 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — BUY (started 2026-02-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 11:00:00 | 1614.50 | 1609.70 | 0.00 | ORB-long ORB[1600.00,1608.00] vol=9.4x ATR=3.41 |
| Stop hit — per-position SL triggered | 2026-02-20 11:05:00 | 1611.09 | 1610.01 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 10:30:00 | 1627.10 | 1616.49 | 0.00 | ORB-long ORB[1603.10,1617.70] vol=1.7x ATR=3.89 |
| Stop hit — per-position SL triggered | 2026-02-23 10:35:00 | 1623.21 | 1616.82 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:55:00 | 1600.30 | 1606.49 | 0.00 | ORB-short ORB[1603.40,1627.00] vol=2.0x ATR=3.41 |
| Stop hit — per-position SL triggered | 2026-02-24 11:30:00 | 1603.71 | 1605.87 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:30:00 | 1597.00 | 1589.80 | 0.00 | ORB-long ORB[1580.10,1592.00] vol=2.3x ATR=3.27 |
| Stop hit — per-position SL triggered | 2026-02-26 11:30:00 | 1593.73 | 1590.61 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 09:45:00 | 1581.60 | 1572.58 | 0.00 | ORB-long ORB[1556.70,1576.80] vol=4.4x ATR=5.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 10:00:00 | 1589.83 | 1574.32 | 0.00 | T1 1.5R @ 1589.83 |
| Target hit | 2026-03-04 15:20:00 | 1633.70 | 1602.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 11:15:00 | 1622.10 | 1628.35 | 0.00 | ORB-short ORB[1622.90,1645.50] vol=3.3x ATR=3.87 |
| Stop hit — per-position SL triggered | 2026-03-13 11:20:00 | 1625.97 | 1628.08 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:10:00 | 1638.40 | 1622.73 | 0.00 | ORB-long ORB[1605.80,1625.00] vol=2.0x ATR=5.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 10:20:00 | 1646.19 | 1628.06 | 0.00 | T1 1.5R @ 1646.19 |
| Stop hit — per-position SL triggered | 2026-03-17 11:50:00 | 1638.40 | 1639.57 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 10:30:00 | 1621.00 | 1614.88 | 0.00 | ORB-long ORB[1600.10,1618.50] vol=3.2x ATR=4.29 |
| Stop hit — per-position SL triggered | 2026-03-18 14:10:00 | 1616.71 | 1617.36 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 11:15:00 | 1458.00 | 1463.41 | 0.00 | ORB-short ORB[1460.30,1480.50] vol=2.5x ATR=2.69 |
| Stop hit — per-position SL triggered | 2026-04-16 11:25:00 | 1460.69 | 1463.23 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:40:00 | 1466.10 | 1463.58 | 0.00 | ORB-long ORB[1455.40,1465.50] vol=1.9x ATR=3.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 09:45:00 | 1471.52 | 1464.99 | 0.00 | T1 1.5R @ 1471.52 |
| Target hit | 2026-04-17 10:50:00 | 1476.30 | 1479.26 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — BUY (started 2026-04-27 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:25:00 | 1493.10 | 1485.24 | 0.00 | ORB-long ORB[1472.20,1490.00] vol=2.4x ATR=4.56 |
| Stop hit — per-position SL triggered | 2026-04-27 11:05:00 | 1488.54 | 1485.89 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:10:00 | 1447.10 | 1455.96 | 0.00 | ORB-short ORB[1454.00,1468.80] vol=3.4x ATR=4.13 |
| Stop hit — per-position SL triggered | 2026-04-30 10:25:00 | 1451.23 | 1455.36 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:40:00 | 1445.00 | 1436.52 | 0.00 | ORB-long ORB[1416.00,1435.30] vol=1.7x ATR=6.10 |
| Stop hit — per-position SL triggered | 2026-05-07 09:45:00 | 1438.90 | 1435.93 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-18 10:15:00 | 1610.10 | 2026-02-18 10:25:00 | 1615.27 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2026-02-18 10:15:00 | 1610.10 | 2026-02-18 11:30:00 | 1612.70 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2026-02-20 11:00:00 | 1614.50 | 2026-02-20 11:05:00 | 1611.09 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-02-23 10:30:00 | 1627.10 | 2026-02-23 10:35:00 | 1623.21 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-24 10:55:00 | 1600.30 | 2026-02-24 11:30:00 | 1603.71 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-02-26 10:30:00 | 1597.00 | 2026-02-26 11:30:00 | 1593.73 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-03-04 09:45:00 | 1581.60 | 2026-03-04 10:00:00 | 1589.83 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-03-04 09:45:00 | 1581.60 | 2026-03-04 15:20:00 | 1633.70 | TARGET_HIT | 0.50 | 3.29% |
| SELL | retest1 | 2026-03-13 11:15:00 | 1622.10 | 2026-03-13 11:20:00 | 1625.97 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-03-17 10:10:00 | 1638.40 | 2026-03-17 10:20:00 | 1646.19 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-03-17 10:10:00 | 1638.40 | 2026-03-17 11:50:00 | 1638.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-18 10:30:00 | 1621.00 | 2026-03-18 14:10:00 | 1616.71 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-16 11:15:00 | 1458.00 | 2026-04-16 11:25:00 | 1460.69 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2026-04-17 09:40:00 | 1466.10 | 2026-04-17 09:45:00 | 1471.52 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-04-17 09:40:00 | 1466.10 | 2026-04-17 10:50:00 | 1476.30 | TARGET_HIT | 0.50 | 0.70% |
| BUY | retest1 | 2026-04-27 10:25:00 | 1493.10 | 2026-04-27 11:05:00 | 1488.54 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-04-30 10:10:00 | 1447.10 | 2026-04-30 10:25:00 | 1451.23 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-05-07 09:40:00 | 1445.00 | 2026-05-07 09:45:00 | 1438.90 | STOP_HIT | 1.00 | -0.42% |
