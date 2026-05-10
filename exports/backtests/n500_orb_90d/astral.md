# Astral Ltd. (ASTRAL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1567.40
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
| ENTRY1 | 13 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 2 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 11
- **Target hits / Stop hits / Partials:** 2 / 11 / 6
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 1.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 6 | 46.2% | 2 | 7 | 4 | 0.09% | 1.2% |
| BUY @ 2nd Alert (retest1) | 13 | 6 | 46.2% | 2 | 7 | 4 | 0.09% | 1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.06% | 0.4% |
| SELL @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.06% | 0.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 8 | 42.1% | 2 | 11 | 6 | 0.08% | 1.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 11:15:00 | 1622.60 | 1610.67 | 0.00 | ORB-long ORB[1589.90,1607.00] vol=2.8x ATR=5.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 12:15:00 | 1630.13 | 1615.61 | 0.00 | T1 1.5R @ 1630.13 |
| Target hit | 2026-02-16 15:20:00 | 1636.60 | 1627.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 11:05:00 | 1629.50 | 1641.36 | 0.00 | ORB-short ORB[1640.40,1654.00] vol=2.0x ATR=3.99 |
| Stop hit — per-position SL triggered | 2026-02-23 11:10:00 | 1633.49 | 1640.59 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 10:50:00 | 1643.20 | 1638.49 | 0.00 | ORB-long ORB[1623.00,1642.60] vol=2.3x ATR=4.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:15:00 | 1649.64 | 1640.98 | 0.00 | T1 1.5R @ 1649.64 |
| Target hit | 2026-02-24 12:30:00 | 1644.20 | 1645.30 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — BUY (started 2026-02-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 11:05:00 | 1692.20 | 1681.20 | 0.00 | ORB-long ORB[1672.60,1686.70] vol=2.4x ATR=3.99 |
| Stop hit — per-position SL triggered | 2026-02-26 11:20:00 | 1688.21 | 1683.40 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:55:00 | 1634.40 | 1647.11 | 0.00 | ORB-short ORB[1643.80,1667.80] vol=2.2x ATR=5.29 |
| Stop hit — per-position SL triggered | 2026-03-05 11:05:00 | 1639.69 | 1646.63 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 09:40:00 | 1688.40 | 1678.06 | 0.00 | ORB-long ORB[1664.10,1681.90] vol=2.2x ATR=5.40 |
| Stop hit — per-position SL triggered | 2026-03-06 09:50:00 | 1683.00 | 1678.95 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:45:00 | 1647.60 | 1640.32 | 0.00 | ORB-long ORB[1619.70,1633.60] vol=1.6x ATR=7.15 |
| Stop hit — per-position SL triggered | 2026-03-17 09:55:00 | 1640.45 | 1640.51 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 11:05:00 | 1569.90 | 1575.71 | 0.00 | ORB-short ORB[1570.90,1590.40] vol=3.7x ATR=6.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-24 11:25:00 | 1560.72 | 1574.69 | 0.00 | T1 1.5R @ 1560.72 |
| Stop hit — per-position SL triggered | 2026-03-24 11:45:00 | 1569.90 | 1574.24 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:50:00 | 1602.40 | 1590.38 | 0.00 | ORB-long ORB[1570.00,1590.70] vol=1.9x ATR=3.86 |
| Stop hit — per-position SL triggered | 2026-04-21 11:25:00 | 1598.54 | 1591.38 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 11:05:00 | 1594.90 | 1587.16 | 0.00 | ORB-long ORB[1579.10,1594.10] vol=1.6x ATR=3.27 |
| Stop hit — per-position SL triggered | 2026-04-22 11:15:00 | 1591.63 | 1587.44 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:10:00 | 1569.90 | 1580.36 | 0.00 | ORB-short ORB[1579.60,1594.50] vol=1.7x ATR=3.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 11:35:00 | 1564.13 | 1579.22 | 0.00 | T1 1.5R @ 1564.13 |
| Stop hit — per-position SL triggered | 2026-04-23 12:10:00 | 1569.90 | 1578.76 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-05-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:45:00 | 1564.00 | 1552.41 | 0.00 | ORB-long ORB[1534.30,1551.90] vol=1.9x ATR=5.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 10:00:00 | 1571.63 | 1556.20 | 0.00 | T1 1.5R @ 1571.63 |
| Stop hit — per-position SL triggered | 2026-05-04 10:20:00 | 1564.00 | 1557.94 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-05-06 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:25:00 | 1547.60 | 1541.39 | 0.00 | ORB-long ORB[1533.10,1547.40] vol=1.7x ATR=3.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:50:00 | 1552.66 | 1545.81 | 0.00 | T1 1.5R @ 1552.66 |
| Stop hit — per-position SL triggered | 2026-05-06 11:00:00 | 1547.60 | 1546.02 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-16 11:15:00 | 1622.60 | 2026-02-16 12:15:00 | 1630.13 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-02-16 11:15:00 | 1622.60 | 2026-02-16 15:20:00 | 1636.60 | TARGET_HIT | 0.50 | 0.86% |
| SELL | retest1 | 2026-02-23 11:05:00 | 1629.50 | 2026-02-23 11:10:00 | 1633.49 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-24 10:50:00 | 1643.20 | 2026-02-24 11:15:00 | 1649.64 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-24 10:50:00 | 1643.20 | 2026-02-24 12:30:00 | 1644.20 | TARGET_HIT | 0.50 | 0.06% |
| BUY | retest1 | 2026-02-26 11:05:00 | 1692.20 | 2026-02-26 11:20:00 | 1688.21 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-03-05 10:55:00 | 1634.40 | 2026-03-05 11:05:00 | 1639.69 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-03-06 09:40:00 | 1688.40 | 2026-03-06 09:50:00 | 1683.00 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-03-17 09:45:00 | 1647.60 | 2026-03-17 09:55:00 | 1640.45 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-03-24 11:05:00 | 1569.90 | 2026-03-24 11:25:00 | 1560.72 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-03-24 11:05:00 | 1569.90 | 2026-03-24 11:45:00 | 1569.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 10:50:00 | 1602.40 | 2026-04-21 11:25:00 | 1598.54 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-04-22 11:05:00 | 1594.90 | 2026-04-22 11:15:00 | 1591.63 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-04-23 11:10:00 | 1569.90 | 2026-04-23 11:35:00 | 1564.13 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-04-23 11:10:00 | 1569.90 | 2026-04-23 12:10:00 | 1569.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-04 09:45:00 | 1564.00 | 2026-05-04 10:00:00 | 1571.63 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-05-04 09:45:00 | 1564.00 | 2026-05-04 10:20:00 | 1564.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-06 10:25:00 | 1547.60 | 2026-05-06 10:50:00 | 1552.66 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-05-06 10:25:00 | 1547.60 | 2026-05-06 11:00:00 | 1547.60 | STOP_HIT | 0.50 | 0.00% |
