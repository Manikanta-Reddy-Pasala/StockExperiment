# BEML Ltd. (BEML)

## Backtest Summary

- **Window:** 2024-01-08 09:15:00 → 2026-05-08 15:25:00 (41842 bars)
- **Last close:** 1952.00
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
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 11
- **Target hits / Stop hits / Partials:** 1 / 11 / 5
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 1.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 3 | 23.1% | 0 | 10 | 3 | -0.13% | -1.7% |
| BUY @ 2nd Alert (retest1) | 13 | 3 | 23.1% | 0 | 10 | 3 | -0.13% | -1.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 3 | 75.0% | 1 | 1 | 2 | 0.77% | 3.1% |
| SELL @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 1 | 2 | 0.77% | 3.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 17 | 6 | 35.3% | 1 | 11 | 5 | 0.08% | 1.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-01-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 09:45:00 | 1604.18 | 1591.66 | 0.00 | ORB-long ORB[1578.48,1599.50] vol=3.1x ATR=8.04 |
| Stop hit — per-position SL triggered | 2024-01-11 09:50:00 | 1596.14 | 1592.30 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-01-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-15 10:55:00 | 1590.50 | 1595.44 | 0.00 | ORB-short ORB[1602.10,1617.43] vol=3.4x ATR=5.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-15 12:10:00 | 1582.52 | 1592.88 | 0.00 | T1 1.5R @ 1582.52 |
| Target hit | 2024-01-15 15:20:00 | 1562.03 | 1584.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2024-01-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-16 10:00:00 | 1581.28 | 1575.34 | 0.00 | ORB-long ORB[1560.00,1579.98] vol=2.1x ATR=6.82 |
| Stop hit — per-position SL triggered | 2024-01-16 10:05:00 | 1574.46 | 1575.31 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-01-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-19 09:50:00 | 1596.58 | 1586.46 | 0.00 | ORB-long ORB[1575.78,1587.35] vol=3.8x ATR=6.08 |
| Stop hit — per-position SL triggered | 2024-01-19 10:00:00 | 1590.50 | 1587.64 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-01-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-23 09:30:00 | 1614.00 | 1607.17 | 0.00 | ORB-long ORB[1593.00,1609.95] vol=5.3x ATR=5.80 |
| Stop hit — per-position SL triggered | 2024-01-23 09:35:00 | 1608.20 | 1607.65 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-20 10:15:00 | 1539.45 | 1555.74 | 0.00 | ORB-short ORB[1555.15,1571.98] vol=1.6x ATR=7.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-20 11:25:00 | 1527.56 | 1548.37 | 0.00 | T1 1.5R @ 1527.56 |
| Stop hit — per-position SL triggered | 2024-02-20 12:05:00 | 1539.45 | 1547.29 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-02-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-23 11:05:00 | 1544.03 | 1532.33 | 0.00 | ORB-long ORB[1525.00,1537.50] vol=3.9x ATR=6.15 |
| Stop hit — per-position SL triggered | 2024-02-23 11:15:00 | 1537.88 | 1533.17 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-02-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-26 09:35:00 | 1537.55 | 1522.69 | 0.00 | ORB-long ORB[1501.53,1523.95] vol=3.2x ATR=9.20 |
| Stop hit — per-position SL triggered | 2024-02-26 09:55:00 | 1528.35 | 1528.30 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-03-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-01 09:35:00 | 1603.55 | 1590.13 | 0.00 | ORB-long ORB[1568.55,1589.00] vol=2.5x ATR=11.93 |
| Stop hit — per-position SL triggered | 2024-03-01 09:40:00 | 1591.62 | 1590.59 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-03-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 10:50:00 | 1527.50 | 1514.01 | 0.00 | ORB-long ORB[1498.80,1521.25] vol=3.0x ATR=5.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-22 11:00:00 | 1536.08 | 1518.38 | 0.00 | T1 1.5R @ 1536.08 |
| Stop hit — per-position SL triggered | 2024-03-22 11:10:00 | 1527.50 | 1519.34 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-04-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 09:30:00 | 1637.10 | 1625.50 | 0.00 | ORB-long ORB[1616.18,1631.93] vol=2.9x ATR=5.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-03 09:40:00 | 1645.24 | 1630.03 | 0.00 | T1 1.5R @ 1645.24 |
| Stop hit — per-position SL triggered | 2024-04-03 09:45:00 | 1637.10 | 1630.63 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-04-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-04 09:35:00 | 1653.53 | 1641.19 | 0.00 | ORB-long ORB[1632.48,1647.08] vol=3.9x ATR=7.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-04 09:40:00 | 1664.92 | 1648.41 | 0.00 | T1 1.5R @ 1664.92 |
| Stop hit — per-position SL triggered | 2024-04-04 09:45:00 | 1653.53 | 1650.82 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-01-11 09:45:00 | 1604.18 | 2024-01-11 09:50:00 | 1596.14 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-01-15 10:55:00 | 1590.50 | 2024-01-15 12:10:00 | 1582.52 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-01-15 10:55:00 | 1590.50 | 2024-01-15 15:20:00 | 1562.03 | TARGET_HIT | 0.50 | 1.79% |
| BUY | retest1 | 2024-01-16 10:00:00 | 1581.28 | 2024-01-16 10:05:00 | 1574.46 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-01-19 09:50:00 | 1596.58 | 2024-01-19 10:00:00 | 1590.50 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-01-23 09:30:00 | 1614.00 | 2024-01-23 09:35:00 | 1608.20 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-02-20 10:15:00 | 1539.45 | 2024-02-20 11:25:00 | 1527.56 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2024-02-20 10:15:00 | 1539.45 | 2024-02-20 12:05:00 | 1539.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-23 11:05:00 | 1544.03 | 2024-02-23 11:15:00 | 1537.88 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-02-26 09:35:00 | 1537.55 | 2024-02-26 09:55:00 | 1528.35 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest1 | 2024-03-01 09:35:00 | 1603.55 | 2024-03-01 09:40:00 | 1591.62 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest1 | 2024-03-22 10:50:00 | 1527.50 | 2024-03-22 11:00:00 | 1536.08 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-03-22 10:50:00 | 1527.50 | 2024-03-22 11:10:00 | 1527.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-03 09:30:00 | 1637.10 | 2024-04-03 09:40:00 | 1645.24 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-04-03 09:30:00 | 1637.10 | 2024-04-03 09:45:00 | 1637.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-04 09:35:00 | 1653.53 | 2024-04-04 09:40:00 | 1664.92 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-04-04 09:35:00 | 1653.53 | 2024-04-04 09:45:00 | 1653.53 | STOP_HIT | 0.50 | 0.00% |
