# R R Kabel Ltd. (RRKABEL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1945.00
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
| TARGET_HIT | 2 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 9
- **Target hits / Stop hits / Partials:** 2 / 9 / 5
- **Avg / median % per leg:** 0.27% / 0.00%
- **Sum % (uncompounded):** 4.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.04% | 0.4% |
| BUY @ 2nd Alert (retest1) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.04% | 0.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.77% | 3.8% |
| SELL @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.77% | 3.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 16 | 7 | 43.8% | 2 | 9 | 5 | 0.27% | 4.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 09:35:00 | 1459.60 | 1473.53 | 0.00 | ORB-short ORB[1469.40,1488.00] vol=2.0x ATR=7.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:10:00 | 1449.01 | 1466.22 | 0.00 | T1 1.5R @ 1449.01 |
| Target hit | 2026-02-10 15:20:00 | 1419.40 | 1436.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2026-02-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:50:00 | 1419.00 | 1409.72 | 0.00 | ORB-long ORB[1396.10,1412.30] vol=4.6x ATR=5.09 |
| Stop hit — per-position SL triggered | 2026-02-17 10:40:00 | 1413.91 | 1411.46 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 10:00:00 | 1433.00 | 1427.27 | 0.00 | ORB-long ORB[1415.50,1426.00] vol=2.4x ATR=5.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:30:00 | 1441.08 | 1431.49 | 0.00 | T1 1.5R @ 1441.08 |
| Target hit | 2026-02-19 11:30:00 | 1442.20 | 1442.94 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — BUY (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:15:00 | 1425.90 | 1422.78 | 0.00 | ORB-long ORB[1407.10,1425.00] vol=1.8x ATR=5.96 |
| Stop hit — per-position SL triggered | 2026-02-20 10:55:00 | 1419.94 | 1422.79 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 09:45:00 | 1471.60 | 1456.87 | 0.00 | ORB-long ORB[1441.00,1454.90] vol=3.6x ATR=6.14 |
| Stop hit — per-position SL triggered | 2026-02-23 09:50:00 | 1465.46 | 1457.86 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:45:00 | 1452.00 | 1478.05 | 0.00 | ORB-short ORB[1492.00,1508.30] vol=1.6x ATR=7.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:05:00 | 1440.87 | 1472.06 | 0.00 | T1 1.5R @ 1440.87 |
| Stop hit — per-position SL triggered | 2026-03-05 11:25:00 | 1452.00 | 1462.72 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 10:15:00 | 1536.70 | 1519.56 | 0.00 | ORB-long ORB[1491.30,1511.70] vol=2.0x ATR=7.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 13:05:00 | 1547.19 | 1532.32 | 0.00 | T1 1.5R @ 1547.19 |
| Stop hit — per-position SL triggered | 2026-03-06 14:40:00 | 1536.70 | 1535.76 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:45:00 | 1387.10 | 1382.22 | 0.00 | ORB-long ORB[1365.70,1386.00] vol=3.6x ATR=5.78 |
| Stop hit — per-position SL triggered | 2026-03-17 09:55:00 | 1381.32 | 1382.25 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:30:00 | 1380.40 | 1378.05 | 0.00 | ORB-long ORB[1366.40,1380.00] vol=2.9x ATR=5.89 |
| Stop hit — per-position SL triggered | 2026-03-25 09:35:00 | 1374.51 | 1377.33 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 09:30:00 | 1461.40 | 1475.43 | 0.00 | ORB-short ORB[1469.20,1489.00] vol=1.7x ATR=5.85 |
| Stop hit — per-position SL triggered | 2026-04-21 09:45:00 | 1467.25 | 1469.58 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:30:00 | 1469.20 | 1456.21 | 0.00 | ORB-long ORB[1445.60,1464.30] vol=2.1x ATR=5.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 10:25:00 | 1477.99 | 1465.62 | 0.00 | T1 1.5R @ 1477.99 |
| Stop hit — per-position SL triggered | 2026-04-22 11:40:00 | 1469.20 | 1470.78 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 09:35:00 | 1459.60 | 2026-02-10 10:10:00 | 1449.01 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2026-02-10 09:35:00 | 1459.60 | 2026-02-10 15:20:00 | 1419.40 | TARGET_HIT | 0.50 | 2.75% |
| BUY | retest1 | 2026-02-17 09:50:00 | 1419.00 | 2026-02-17 10:40:00 | 1413.91 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-02-19 10:00:00 | 1433.00 | 2026-02-19 10:30:00 | 1441.08 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-02-19 10:00:00 | 1433.00 | 2026-02-19 11:30:00 | 1442.20 | TARGET_HIT | 0.50 | 0.64% |
| BUY | retest1 | 2026-02-20 10:15:00 | 1425.90 | 2026-02-20 10:55:00 | 1419.94 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-02-23 09:45:00 | 1471.60 | 2026-02-23 09:50:00 | 1465.46 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-03-05 10:45:00 | 1452.00 | 2026-03-05 11:05:00 | 1440.87 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2026-03-05 10:45:00 | 1452.00 | 2026-03-05 11:25:00 | 1452.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-06 10:15:00 | 1536.70 | 2026-03-06 13:05:00 | 1547.19 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2026-03-06 10:15:00 | 1536.70 | 2026-03-06 14:40:00 | 1536.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 09:45:00 | 1387.10 | 2026-03-17 09:55:00 | 1381.32 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-03-25 09:30:00 | 1380.40 | 2026-03-25 09:35:00 | 1374.51 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-04-21 09:30:00 | 1461.40 | 2026-04-21 09:45:00 | 1467.25 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-22 09:30:00 | 1469.20 | 2026-04-22 10:25:00 | 1477.99 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-04-22 09:30:00 | 1469.20 | 2026-04-22 11:40:00 | 1469.20 | STOP_HIT | 0.50 | 0.00% |
