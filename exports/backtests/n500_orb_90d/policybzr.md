# PB Fintech Ltd. (POLICYBZR)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1647.90
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
| TARGET_HIT | 3 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 12
- **Target hits / Stop hits / Partials:** 3 / 12 / 6
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 2.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 3 | 27.3% | 1 | 8 | 2 | -0.00% | -0.0% |
| BUY @ 2nd Alert (retest1) | 11 | 3 | 27.3% | 1 | 8 | 2 | -0.00% | -0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 6 | 60.0% | 2 | 4 | 4 | 0.24% | 2.4% |
| SELL @ 2nd Alert (retest1) | 10 | 6 | 60.0% | 2 | 4 | 4 | 0.24% | 2.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 21 | 9 | 42.9% | 3 | 12 | 6 | 0.11% | 2.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 11:15:00 | 1486.00 | 1493.65 | 0.00 | ORB-short ORB[1493.40,1514.80] vol=1.6x ATR=5.29 |
| Stop hit — per-position SL triggered | 2026-02-17 11:55:00 | 1491.29 | 1492.81 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:35:00 | 1496.30 | 1495.06 | 0.00 | ORB-long ORB[1484.00,1496.00] vol=8.5x ATR=4.85 |
| Stop hit — per-position SL triggered | 2026-02-19 09:45:00 | 1491.45 | 1494.72 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:50:00 | 1531.00 | 1519.82 | 0.00 | ORB-long ORB[1510.60,1517.90] vol=1.5x ATR=4.52 |
| Stop hit — per-position SL triggered | 2026-02-26 11:00:00 | 1526.48 | 1520.91 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:00:00 | 1495.80 | 1482.55 | 0.00 | ORB-long ORB[1465.60,1483.90] vol=2.4x ATR=6.42 |
| Stop hit — per-position SL triggered | 2026-03-17 11:15:00 | 1489.38 | 1490.89 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 10:50:00 | 1522.50 | 1511.16 | 0.00 | ORB-long ORB[1496.00,1517.50] vol=1.9x ATR=6.13 |
| Stop hit — per-position SL triggered | 2026-03-20 11:35:00 | 1516.37 | 1514.32 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 10:55:00 | 1451.00 | 1461.88 | 0.00 | ORB-short ORB[1465.70,1485.00] vol=5.0x ATR=5.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 11:40:00 | 1442.46 | 1454.56 | 0.00 | T1 1.5R @ 1442.46 |
| Target hit | 2026-03-23 15:20:00 | 1437.20 | 1440.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2026-03-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:35:00 | 1443.00 | 1449.59 | 0.00 | ORB-short ORB[1450.00,1464.70] vol=1.9x ATR=4.85 |
| Stop hit — per-position SL triggered | 2026-03-24 10:45:00 | 1447.85 | 1449.55 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 10:20:00 | 1505.00 | 1496.43 | 0.00 | ORB-long ORB[1482.20,1494.90] vol=6.0x ATR=6.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 10:25:00 | 1514.60 | 1498.59 | 0.00 | T1 1.5R @ 1514.60 |
| Stop hit — per-position SL triggered | 2026-04-16 11:50:00 | 1505.00 | 1500.96 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:00:00 | 1633.30 | 1618.84 | 0.00 | ORB-long ORB[1603.30,1623.20] vol=1.8x ATR=6.34 |
| Stop hit — per-position SL triggered | 2026-04-22 10:15:00 | 1626.96 | 1620.03 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:15:00 | 1647.50 | 1636.35 | 0.00 | ORB-long ORB[1610.50,1633.50] vol=5.8x ATR=5.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 10:25:00 | 1655.62 | 1637.67 | 0.00 | T1 1.5R @ 1655.62 |
| Target hit | 2026-04-23 15:20:00 | 1670.30 | 1661.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2026-04-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:05:00 | 1721.80 | 1714.22 | 0.00 | ORB-long ORB[1696.00,1714.00] vol=3.0x ATR=6.16 |
| Stop hit — per-position SL triggered | 2026-04-27 10:25:00 | 1715.64 | 1716.46 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:00:00 | 1659.30 | 1668.56 | 0.00 | ORB-short ORB[1664.90,1680.10] vol=1.9x ATR=6.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 10:15:00 | 1650.21 | 1666.42 | 0.00 | T1 1.5R @ 1650.21 |
| Target hit | 2026-04-30 11:50:00 | 1658.90 | 1658.86 | 0.00 | Trail-exit close>VWAP |

### Cycle 13 — SELL (started 2026-05-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 10:25:00 | 1663.30 | 1674.72 | 0.00 | ORB-short ORB[1667.70,1682.90] vol=1.8x ATR=5.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 10:40:00 | 1655.47 | 1670.98 | 0.00 | T1 1.5R @ 1655.47 |
| Stop hit — per-position SL triggered | 2026-05-04 10:55:00 | 1663.30 | 1670.02 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 10:10:00 | 1680.00 | 1668.76 | 0.00 | ORB-long ORB[1647.40,1665.00] vol=2.8x ATR=6.09 |
| Stop hit — per-position SL triggered | 2026-05-05 10:25:00 | 1673.91 | 1670.56 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-05-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:05:00 | 1676.60 | 1688.25 | 0.00 | ORB-short ORB[1681.60,1703.00] vol=1.9x ATR=6.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:10:00 | 1667.59 | 1684.39 | 0.00 | T1 1.5R @ 1667.59 |
| Stop hit — per-position SL triggered | 2026-05-06 10:15:00 | 1676.60 | 1683.51 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-17 11:15:00 | 1486.00 | 2026-02-17 11:55:00 | 1491.29 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-02-19 09:35:00 | 1496.30 | 2026-02-19 09:45:00 | 1491.45 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-26 10:50:00 | 1531.00 | 2026-02-26 11:00:00 | 1526.48 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-03-17 10:00:00 | 1495.80 | 2026-03-17 11:15:00 | 1489.38 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-03-20 10:50:00 | 1522.50 | 2026-03-20 11:35:00 | 1516.37 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-03-23 10:55:00 | 1451.00 | 2026-03-23 11:40:00 | 1442.46 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-03-23 10:55:00 | 1451.00 | 2026-03-23 15:20:00 | 1437.20 | TARGET_HIT | 0.50 | 0.95% |
| SELL | retest1 | 2026-03-24 10:35:00 | 1443.00 | 2026-03-24 10:45:00 | 1447.85 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-16 10:20:00 | 1505.00 | 2026-04-16 10:25:00 | 1514.60 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2026-04-16 10:20:00 | 1505.00 | 2026-04-16 11:50:00 | 1505.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 10:00:00 | 1633.30 | 2026-04-22 10:15:00 | 1626.96 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-04-23 10:15:00 | 1647.50 | 2026-04-23 10:25:00 | 1655.62 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-04-23 10:15:00 | 1647.50 | 2026-04-23 15:20:00 | 1670.30 | TARGET_HIT | 0.50 | 1.38% |
| BUY | retest1 | 2026-04-27 10:05:00 | 1721.80 | 2026-04-27 10:25:00 | 1715.64 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-04-30 10:00:00 | 1659.30 | 2026-04-30 10:15:00 | 1650.21 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-04-30 10:00:00 | 1659.30 | 2026-04-30 11:50:00 | 1658.90 | TARGET_HIT | 0.50 | 0.02% |
| SELL | retest1 | 2026-05-04 10:25:00 | 1663.30 | 2026-05-04 10:40:00 | 1655.47 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-05-04 10:25:00 | 1663.30 | 2026-05-04 10:55:00 | 1663.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-05 10:10:00 | 1680.00 | 2026-05-05 10:25:00 | 1673.91 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-05-06 10:05:00 | 1676.60 | 2026-05-06 10:10:00 | 1667.59 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-05-06 10:05:00 | 1676.60 | 2026-05-06 10:15:00 | 1676.60 | STOP_HIT | 0.50 | 0.00% |
