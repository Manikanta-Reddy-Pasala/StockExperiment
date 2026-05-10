# Gravita India Ltd. (GRAVITA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1760.60
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
| ENTRY1 | 12 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 9
- **Target hits / Stop hits / Partials:** 3 / 9 / 6
- **Avg / median % per leg:** 0.21% / 0.02%
- **Sum % (uncompounded):** 3.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 6 | 42.9% | 2 | 8 | 4 | 0.07% | 0.9% |
| BUY @ 2nd Alert (retest1) | 14 | 6 | 42.9% | 2 | 8 | 4 | 0.07% | 0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 3 | 75.0% | 1 | 1 | 2 | 0.71% | 2.8% |
| SELL @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 1 | 2 | 0.71% | 2.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 18 | 9 | 50.0% | 3 | 9 | 6 | 0.21% | 3.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:35:00 | 1662.30 | 1653.36 | 0.00 | ORB-long ORB[1640.00,1661.20] vol=2.6x ATR=8.07 |
| Stop hit — per-position SL triggered | 2026-02-17 10:00:00 | 1654.23 | 1656.01 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:55:00 | 1646.10 | 1651.99 | 0.00 | ORB-short ORB[1647.30,1662.50] vol=1.9x ATR=5.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:10:00 | 1638.45 | 1648.09 | 0.00 | T1 1.5R @ 1638.45 |
| Target hit | 2026-02-18 15:20:00 | 1616.50 | 1629.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-02-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:50:00 | 1641.30 | 1634.67 | 0.00 | ORB-long ORB[1619.50,1634.00] vol=1.8x ATR=5.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:15:00 | 1649.32 | 1638.37 | 0.00 | T1 1.5R @ 1649.32 |
| Stop hit — per-position SL triggered | 2026-02-19 10:25:00 | 1641.30 | 1639.26 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 09:55:00 | 1585.10 | 1591.87 | 0.00 | ORB-short ORB[1593.80,1604.00] vol=1.9x ATR=6.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:05:00 | 1575.88 | 1588.18 | 0.00 | T1 1.5R @ 1575.88 |
| Stop hit — per-position SL triggered | 2026-02-23 15:00:00 | 1585.10 | 1580.92 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:15:00 | 1599.40 | 1593.32 | 0.00 | ORB-long ORB[1579.80,1598.40] vol=1.9x ATR=3.57 |
| Stop hit — per-position SL triggered | 2026-02-25 10:30:00 | 1595.83 | 1593.70 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:30:00 | 1642.00 | 1621.84 | 0.00 | ORB-long ORB[1598.00,1619.90] vol=5.8x ATR=6.67 |
| Stop hit — per-position SL triggered | 2026-02-26 09:35:00 | 1635.33 | 1630.14 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 10:35:00 | 1545.90 | 1536.53 | 0.00 | ORB-long ORB[1530.00,1543.80] vol=2.4x ATR=5.90 |
| Stop hit — per-position SL triggered | 2026-03-05 11:20:00 | 1540.00 | 1537.21 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:30:00 | 1530.20 | 1518.15 | 0.00 | ORB-long ORB[1500.70,1518.50] vol=4.7x ATR=6.28 |
| Stop hit — per-position SL triggered | 2026-03-11 09:35:00 | 1523.92 | 1519.21 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 1427.80 | 1421.39 | 0.00 | ORB-long ORB[1409.80,1426.10] vol=1.7x ATR=4.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 09:45:00 | 1434.55 | 1424.93 | 0.00 | T1 1.5R @ 1434.55 |
| Stop hit — per-position SL triggered | 2026-03-18 09:55:00 | 1427.80 | 1425.53 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:20:00 | 1542.30 | 1535.62 | 0.00 | ORB-long ORB[1516.80,1538.00] vol=1.5x ATR=6.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 13:05:00 | 1551.89 | 1541.10 | 0.00 | T1 1.5R @ 1551.89 |
| Target hit | 2026-04-10 14:20:00 | 1542.60 | 1542.77 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — BUY (started 2026-04-15 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:20:00 | 1605.00 | 1591.93 | 0.00 | ORB-long ORB[1579.00,1599.70] vol=3.2x ATR=7.20 |
| Stop hit — per-position SL triggered | 2026-04-15 10:50:00 | 1597.80 | 1594.52 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 11:15:00 | 1636.50 | 1622.74 | 0.00 | ORB-long ORB[1610.50,1629.00] vol=3.3x ATR=4.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 11:20:00 | 1643.44 | 1628.74 | 0.00 | T1 1.5R @ 1643.44 |
| Target hit | 2026-04-29 14:20:00 | 1657.00 | 1657.40 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-17 09:35:00 | 1662.30 | 2026-02-17 10:00:00 | 1654.23 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-02-18 09:55:00 | 1646.10 | 2026-02-18 10:10:00 | 1638.45 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-02-18 09:55:00 | 1646.10 | 2026-02-18 15:20:00 | 1616.50 | TARGET_HIT | 0.50 | 1.80% |
| BUY | retest1 | 2026-02-19 09:50:00 | 1641.30 | 2026-02-19 10:15:00 | 1649.32 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-02-19 09:50:00 | 1641.30 | 2026-02-19 10:25:00 | 1641.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-23 09:55:00 | 1585.10 | 2026-02-23 11:05:00 | 1575.88 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-02-23 09:55:00 | 1585.10 | 2026-02-23 15:00:00 | 1585.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 10:15:00 | 1599.40 | 2026-02-25 10:30:00 | 1595.83 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-26 09:30:00 | 1642.00 | 2026-02-26 09:35:00 | 1635.33 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-03-05 10:35:00 | 1545.90 | 2026-03-05 11:20:00 | 1540.00 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-03-11 09:30:00 | 1530.20 | 2026-03-11 09:35:00 | 1523.92 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-03-18 09:30:00 | 1427.80 | 2026-03-18 09:45:00 | 1434.55 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-03-18 09:30:00 | 1427.80 | 2026-03-18 09:55:00 | 1427.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-10 10:20:00 | 1542.30 | 2026-04-10 13:05:00 | 1551.89 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-04-10 10:20:00 | 1542.30 | 2026-04-10 14:20:00 | 1542.60 | TARGET_HIT | 0.50 | 0.02% |
| BUY | retest1 | 2026-04-15 10:20:00 | 1605.00 | 2026-04-15 10:50:00 | 1597.80 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-04-29 11:15:00 | 1636.50 | 2026-04-29 11:20:00 | 1643.44 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-04-29 11:15:00 | 1636.50 | 2026-04-29 14:20:00 | 1657.00 | TARGET_HIT | 0.50 | 1.25% |
