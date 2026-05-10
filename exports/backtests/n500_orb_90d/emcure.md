# Emcure Pharmaceuticals Ltd. (EMCURE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1646.00
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
| TARGET_HIT | 3 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 8
- **Target hits / Stop hits / Partials:** 3 / 8 / 5
- **Avg / median % per leg:** 0.10% / 0.21%
- **Sum % (uncompounded):** 1.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 4 | 44.4% | 1 | 5 | 3 | 0.08% | 0.7% |
| BUY @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 1 | 5 | 3 | 0.08% | 0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 4 | 57.1% | 2 | 3 | 2 | 0.14% | 1.0% |
| SELL @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 2 | 3 | 2 | 0.14% | 1.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 16 | 8 | 50.0% | 3 | 8 | 5 | 0.10% | 1.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 09:40:00 | 1476.50 | 1486.98 | 0.00 | ORB-short ORB[1483.80,1496.80] vol=1.9x ATR=5.04 |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 1481.54 | 1481.80 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:45:00 | 1447.80 | 1435.25 | 0.00 | ORB-long ORB[1424.10,1444.40] vol=1.8x ATR=5.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 09:50:00 | 1455.64 | 1445.13 | 0.00 | T1 1.5R @ 1455.64 |
| Target hit | 2026-02-26 11:05:00 | 1450.80 | 1454.19 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — SELL (started 2026-03-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:25:00 | 1464.70 | 1474.17 | 0.00 | ORB-short ORB[1467.10,1482.00] vol=2.1x ATR=6.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:00:00 | 1454.87 | 1470.39 | 0.00 | T1 1.5R @ 1454.87 |
| Target hit | 2026-03-05 15:20:00 | 1455.00 | 1453.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2026-03-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 09:35:00 | 1469.90 | 1458.84 | 0.00 | ORB-long ORB[1447.90,1462.70] vol=1.9x ATR=6.68 |
| Stop hit — per-position SL triggered | 2026-03-06 10:10:00 | 1463.22 | 1464.76 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 09:50:00 | 1494.80 | 1504.44 | 0.00 | ORB-short ORB[1498.30,1520.00] vol=2.2x ATR=7.14 |
| Stop hit — per-position SL triggered | 2026-03-19 10:00:00 | 1501.94 | 1504.21 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 10:30:00 | 1483.10 | 1496.94 | 0.00 | ORB-short ORB[1485.00,1505.90] vol=2.8x ATR=6.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 12:20:00 | 1473.23 | 1489.32 | 0.00 | T1 1.5R @ 1473.23 |
| Target hit | 2026-03-20 15:20:00 | 1479.50 | 1485.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2026-03-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:30:00 | 1570.10 | 1559.70 | 0.00 | ORB-long ORB[1545.20,1565.60] vol=3.0x ATR=6.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 10:00:00 | 1580.51 | 1578.98 | 0.00 | T1 1.5R @ 1580.51 |
| Stop hit — per-position SL triggered | 2026-03-25 10:10:00 | 1570.10 | 1584.30 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 09:35:00 | 1582.80 | 1574.39 | 0.00 | ORB-long ORB[1566.10,1581.00] vol=2.0x ATR=5.09 |
| Stop hit — per-position SL triggered | 2026-04-09 09:40:00 | 1577.71 | 1575.46 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-22 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:05:00 | 1609.00 | 1601.38 | 0.00 | ORB-long ORB[1591.90,1604.60] vol=1.8x ATR=5.13 |
| Stop hit — per-position SL triggered | 2026-04-22 10:40:00 | 1603.87 | 1603.42 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:35:00 | 1612.30 | 1602.22 | 0.00 | ORB-long ORB[1591.50,1607.90] vol=2.1x ATR=4.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 09:45:00 | 1618.65 | 1608.72 | 0.00 | T1 1.5R @ 1618.65 |
| Stop hit — per-position SL triggered | 2026-04-23 09:55:00 | 1612.30 | 1610.33 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-05-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:40:00 | 1639.50 | 1652.85 | 0.00 | ORB-short ORB[1651.30,1675.00] vol=4.2x ATR=7.73 |
| Stop hit — per-position SL triggered | 2026-05-08 10:10:00 | 1647.23 | 1647.85 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-12 09:40:00 | 1476.50 | 2026-02-12 10:15:00 | 1481.54 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-26 09:45:00 | 1447.80 | 2026-02-26 09:50:00 | 1455.64 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-02-26 09:45:00 | 1447.80 | 2026-02-26 11:05:00 | 1450.80 | TARGET_HIT | 0.50 | 0.21% |
| SELL | retest1 | 2026-03-05 10:25:00 | 1464.70 | 2026-03-05 11:00:00 | 1454.87 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2026-03-05 10:25:00 | 1464.70 | 2026-03-05 15:20:00 | 1455.00 | TARGET_HIT | 0.50 | 0.66% |
| BUY | retest1 | 2026-03-06 09:35:00 | 1469.90 | 2026-03-06 10:10:00 | 1463.22 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-03-19 09:50:00 | 1494.80 | 2026-03-19 10:00:00 | 1501.94 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2026-03-20 10:30:00 | 1483.10 | 2026-03-20 12:20:00 | 1473.23 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2026-03-20 10:30:00 | 1483.10 | 2026-03-20 15:20:00 | 1479.50 | TARGET_HIT | 0.50 | 0.24% |
| BUY | retest1 | 2026-03-25 09:30:00 | 1570.10 | 2026-03-25 10:00:00 | 1580.51 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-03-25 09:30:00 | 1570.10 | 2026-03-25 10:10:00 | 1570.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-09 09:35:00 | 1582.80 | 2026-04-09 09:40:00 | 1577.71 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-22 10:05:00 | 1609.00 | 2026-04-22 10:40:00 | 1603.87 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-23 09:35:00 | 1612.30 | 2026-04-23 09:45:00 | 1618.65 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-04-23 09:35:00 | 1612.30 | 2026-04-23 09:55:00 | 1612.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-08 09:40:00 | 1639.50 | 2026-05-08 10:10:00 | 1647.23 | STOP_HIT | 1.00 | -0.47% |
