# Poly Medicure Ltd. (POLYMED)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1649.50
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
| ENTRY1 | 11 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 4 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 7
- **Target hits / Stop hits / Partials:** 4 / 7 / 5
- **Avg / median % per leg:** 0.19% / 0.23%
- **Sum % (uncompounded):** 3.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 7 | 63.6% | 3 | 4 | 4 | 0.21% | 2.3% |
| BUY @ 2nd Alert (retest1) | 11 | 7 | 63.6% | 3 | 4 | 4 | 0.21% | 2.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.16% | 0.8% |
| SELL @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.16% | 0.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 16 | 9 | 56.2% | 4 | 7 | 5 | 0.19% | 3.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:05:00 | 1296.70 | 1301.37 | 0.00 | ORB-short ORB[1297.10,1315.90] vol=2.6x ATR=3.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:40:00 | 1291.45 | 1300.00 | 0.00 | T1 1.5R @ 1291.45 |
| Target hit | 2026-02-19 15:20:00 | 1276.70 | 1288.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:55:00 | 1285.00 | 1289.42 | 0.00 | ORB-short ORB[1285.10,1299.60] vol=1.8x ATR=4.58 |
| Stop hit — per-position SL triggered | 2026-02-26 11:15:00 | 1289.58 | 1289.15 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-03-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-24 09:30:00 | 1244.60 | 1231.85 | 0.00 | ORB-long ORB[1213.00,1230.00] vol=5.0x ATR=8.16 |
| Stop hit — per-position SL triggered | 2026-03-24 09:35:00 | 1236.44 | 1232.74 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-04-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 09:30:00 | 1455.50 | 1464.41 | 0.00 | ORB-short ORB[1460.00,1480.00] vol=3.3x ATR=6.08 |
| Stop hit — per-position SL triggered | 2026-04-15 09:40:00 | 1461.58 | 1462.06 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-04-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:40:00 | 1465.00 | 1474.53 | 0.00 | ORB-short ORB[1472.60,1485.00] vol=1.6x ATR=5.42 |
| Stop hit — per-position SL triggered | 2026-04-16 10:40:00 | 1470.42 | 1468.98 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:40:00 | 1490.00 | 1466.33 | 0.00 | ORB-long ORB[1450.70,1462.80] vol=1.9x ATR=6.97 |
| Stop hit — per-position SL triggered | 2026-04-21 09:50:00 | 1483.03 | 1473.61 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:35:00 | 1484.20 | 1474.64 | 0.00 | ORB-long ORB[1460.00,1480.00] vol=7.7x ATR=6.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 09:40:00 | 1494.41 | 1477.08 | 0.00 | T1 1.5R @ 1494.41 |
| Target hit | 2026-04-23 12:30:00 | 1491.50 | 1492.08 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — BUY (started 2026-04-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:55:00 | 1504.00 | 1490.00 | 0.00 | ORB-long ORB[1478.00,1498.30] vol=2.1x ATR=9.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 12:55:00 | 1518.07 | 1503.19 | 0.00 | T1 1.5R @ 1518.07 |
| Target hit | 2026-04-28 14:10:00 | 1507.40 | 1507.56 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2026-05-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:40:00 | 1551.00 | 1534.08 | 0.00 | ORB-long ORB[1521.40,1531.90] vol=2.1x ATR=8.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 09:50:00 | 1563.67 | 1548.27 | 0.00 | T1 1.5R @ 1563.67 |
| Target hit | 2026-05-05 10:25:00 | 1554.00 | 1558.68 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2026-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:30:00 | 1644.60 | 1636.81 | 0.00 | ORB-long ORB[1620.00,1642.00] vol=1.7x ATR=10.39 |
| Stop hit — per-position SL triggered | 2026-05-06 10:05:00 | 1634.21 | 1641.25 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-05-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 11:00:00 | 1678.90 | 1669.00 | 0.00 | ORB-long ORB[1662.80,1677.70] vol=1.6x ATR=7.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 11:15:00 | 1690.39 | 1671.69 | 0.00 | T1 1.5R @ 1690.39 |
| Stop hit — per-position SL triggered | 2026-05-07 12:05:00 | 1678.90 | 1674.02 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-19 11:05:00 | 1296.70 | 2026-02-19 11:40:00 | 1291.45 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-02-19 11:05:00 | 1296.70 | 2026-02-19 15:20:00 | 1276.70 | TARGET_HIT | 0.50 | 1.54% |
| SELL | retest1 | 2026-02-26 10:55:00 | 1285.00 | 2026-02-26 11:15:00 | 1289.58 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-03-24 09:30:00 | 1244.60 | 2026-03-24 09:35:00 | 1236.44 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest1 | 2026-04-15 09:30:00 | 1455.50 | 2026-04-15 09:40:00 | 1461.58 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-04-16 09:40:00 | 1465.00 | 2026-04-16 10:40:00 | 1470.42 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-21 09:40:00 | 1490.00 | 2026-04-21 09:50:00 | 1483.03 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-04-23 09:35:00 | 1484.20 | 2026-04-23 09:40:00 | 1494.41 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2026-04-23 09:35:00 | 1484.20 | 2026-04-23 12:30:00 | 1491.50 | TARGET_HIT | 0.50 | 0.49% |
| BUY | retest1 | 2026-04-28 09:55:00 | 1504.00 | 2026-04-28 12:55:00 | 1518.07 | PARTIAL | 0.50 | 0.94% |
| BUY | retest1 | 2026-04-28 09:55:00 | 1504.00 | 2026-04-28 14:10:00 | 1507.40 | TARGET_HIT | 0.50 | 0.23% |
| BUY | retest1 | 2026-05-05 09:40:00 | 1551.00 | 2026-05-05 09:50:00 | 1563.67 | PARTIAL | 0.50 | 0.82% |
| BUY | retest1 | 2026-05-05 09:40:00 | 1551.00 | 2026-05-05 10:25:00 | 1554.00 | TARGET_HIT | 0.50 | 0.19% |
| BUY | retest1 | 2026-05-06 09:30:00 | 1644.60 | 2026-05-06 10:05:00 | 1634.21 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest1 | 2026-05-07 11:00:00 | 1678.90 | 2026-05-07 11:15:00 | 1690.39 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2026-05-07 11:00:00 | 1678.90 | 2026-05-07 12:05:00 | 1678.90 | STOP_HIT | 0.50 | 0.00% |
