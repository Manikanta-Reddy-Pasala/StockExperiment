# Tata Communications Ltd. (TATACOMM)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1582.60
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
| ENTRY1 | 8 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 7
- **Target hits / Stop hits / Partials:** 1 / 7 / 1
- **Avg / median % per leg:** -0.08% / -0.34%
- **Sum % (uncompounded):** -0.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.39% | -1.2% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.39% | -1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 2 | 33.3% | 1 | 4 | 1 | 0.07% | 0.4% |
| SELL @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 1 | 4 | 1 | 0.07% | 0.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 2 | 22.2% | 1 | 7 | 1 | -0.08% | -0.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:00:00 | 1604.10 | 1594.13 | 0.00 | ORB-long ORB[1580.00,1599.00] vol=2.1x ATR=5.91 |
| Stop hit — per-position SL triggered | 2026-02-10 10:05:00 | 1598.19 | 1597.38 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-03-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 11:00:00 | 1487.30 | 1474.97 | 0.00 | ORB-long ORB[1439.80,1454.00] vol=2.3x ATR=6.52 |
| Stop hit — per-position SL triggered | 2026-03-06 11:05:00 | 1480.78 | 1475.18 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-03-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 10:10:00 | 1491.00 | 1481.80 | 0.00 | ORB-long ORB[1466.00,1488.00] vol=1.9x ATR=5.48 |
| Stop hit — per-position SL triggered | 2026-03-11 11:35:00 | 1485.52 | 1488.17 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:40:00 | 1418.00 | 1427.75 | 0.00 | ORB-short ORB[1432.40,1452.00] vol=2.4x ATR=5.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 11:10:00 | 1410.45 | 1418.57 | 0.00 | T1 1.5R @ 1410.45 |
| Target hit | 2026-03-13 15:20:00 | 1399.10 | 1413.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2026-03-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:55:00 | 1383.90 | 1395.31 | 0.00 | ORB-short ORB[1393.30,1407.90] vol=1.5x ATR=4.70 |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 1388.60 | 1393.74 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 11:05:00 | 1376.80 | 1400.71 | 0.00 | ORB-short ORB[1414.10,1434.00] vol=1.7x ATR=6.33 |
| Stop hit — per-position SL triggered | 2026-03-24 11:30:00 | 1383.13 | 1396.07 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:55:00 | 1583.70 | 1597.29 | 0.00 | ORB-short ORB[1599.00,1612.60] vol=2.3x ATR=4.99 |
| Stop hit — per-position SL triggered | 2026-04-29 11:10:00 | 1588.69 | 1596.08 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-05-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 10:50:00 | 1557.70 | 1577.95 | 0.00 | ORB-short ORB[1576.00,1597.40] vol=2.4x ATR=5.23 |
| Stop hit — per-position SL triggered | 2026-05-04 11:00:00 | 1562.93 | 1575.77 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 10:00:00 | 1604.10 | 2026-02-10 10:05:00 | 1598.19 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-03-06 11:00:00 | 1487.30 | 2026-03-06 11:05:00 | 1480.78 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-03-11 10:10:00 | 1491.00 | 2026-03-11 11:35:00 | 1485.52 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-03-13 10:40:00 | 1418.00 | 2026-03-13 11:10:00 | 1410.45 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-03-13 10:40:00 | 1418.00 | 2026-03-13 15:20:00 | 1399.10 | TARGET_HIT | 0.50 | 1.33% |
| SELL | retest1 | 2026-03-16 10:55:00 | 1383.90 | 2026-03-16 11:15:00 | 1388.60 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-24 11:05:00 | 1376.80 | 2026-03-24 11:30:00 | 1383.13 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-04-29 10:55:00 | 1583.70 | 2026-04-29 11:10:00 | 1588.69 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-05-04 10:50:00 | 1557.70 | 2026-05-04 11:00:00 | 1562.93 | STOP_HIT | 1.00 | -0.34% |
