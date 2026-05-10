# Dr. Lal Path Labs Ltd. (LALPATHLAB)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1655.00
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
| ENTRY1 | 10 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 8
- **Target hits / Stop hits / Partials:** 2 / 8 / 5
- **Avg / median % per leg:** 0.23% / 0.00%
- **Sum % (uncompounded):** 3.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 7 | 53.8% | 2 | 6 | 5 | 0.31% | 4.1% |
| BUY @ 2nd Alert (retest1) | 13 | 7 | 53.8% | 2 | 6 | 5 | 0.31% | 4.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.29% | -0.6% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.29% | -0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 15 | 7 | 46.7% | 2 | 8 | 5 | 0.23% | 3.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:45:00 | 1459.80 | 1447.52 | 0.00 | ORB-long ORB[1440.10,1452.10] vol=1.8x ATR=6.54 |
| Stop hit — per-position SL triggered | 2026-02-10 11:25:00 | 1453.26 | 1450.00 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:35:00 | 1383.80 | 1388.99 | 0.00 | ORB-short ORB[1386.40,1396.00] vol=1.6x ATR=5.31 |
| Stop hit — per-position SL triggered | 2026-02-24 09:55:00 | 1389.11 | 1387.96 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:50:00 | 1421.40 | 1416.33 | 0.00 | ORB-long ORB[1407.40,1415.90] vol=1.7x ATR=3.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:45:00 | 1427.17 | 1418.67 | 0.00 | T1 1.5R @ 1427.17 |
| Stop hit — per-position SL triggered | 2026-02-25 12:25:00 | 1421.40 | 1419.52 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 11:15:00 | 1400.90 | 1407.97 | 0.00 | ORB-short ORB[1405.00,1425.00] vol=4.2x ATR=2.74 |
| Stop hit — per-position SL triggered | 2026-02-26 11:20:00 | 1403.64 | 1407.72 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 10:30:00 | 1356.50 | 1354.76 | 0.00 | ORB-long ORB[1332.50,1351.90] vol=7.1x ATR=4.23 |
| Stop hit — per-position SL triggered | 2026-03-06 13:00:00 | 1352.27 | 1355.27 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:55:00 | 1393.40 | 1385.35 | 0.00 | ORB-long ORB[1368.00,1384.30] vol=2.7x ATR=4.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 10:05:00 | 1399.80 | 1394.16 | 0.00 | T1 1.5R @ 1399.80 |
| Target hit | 2026-03-11 15:20:00 | 1402.00 | 1402.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2026-04-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 11:10:00 | 1411.50 | 1407.41 | 0.00 | ORB-long ORB[1389.00,1409.90] vol=15.1x ATR=3.76 |
| Stop hit — per-position SL triggered | 2026-04-10 11:45:00 | 1407.74 | 1407.96 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:50:00 | 1406.90 | 1396.35 | 0.00 | ORB-long ORB[1385.10,1403.90] vol=1.7x ATR=6.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 10:00:00 | 1416.13 | 1405.32 | 0.00 | T1 1.5R @ 1416.13 |
| Target hit | 2026-04-15 13:55:00 | 1436.10 | 1437.16 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2026-04-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:55:00 | 1423.60 | 1421.43 | 0.00 | ORB-long ORB[1406.80,1421.80] vol=11.7x ATR=3.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 11:05:00 | 1429.52 | 1423.08 | 0.00 | T1 1.5R @ 1429.52 |
| Stop hit — per-position SL triggered | 2026-04-21 11:55:00 | 1423.60 | 1425.39 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:40:00 | 1461.50 | 1455.93 | 0.00 | ORB-long ORB[1442.00,1458.20] vol=5.7x ATR=4.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 11:00:00 | 1468.56 | 1457.39 | 0.00 | T1 1.5R @ 1468.56 |
| Stop hit — per-position SL triggered | 2026-04-23 11:10:00 | 1461.50 | 1457.93 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 10:45:00 | 1459.80 | 2026-02-10 11:25:00 | 1453.26 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-02-24 09:35:00 | 1383.80 | 2026-02-24 09:55:00 | 1389.11 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-02-25 10:50:00 | 1421.40 | 2026-02-25 11:45:00 | 1427.17 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-02-25 10:50:00 | 1421.40 | 2026-02-25 12:25:00 | 1421.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-26 11:15:00 | 1400.90 | 2026-02-26 11:20:00 | 1403.64 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-03-06 10:30:00 | 1356.50 | 2026-03-06 13:00:00 | 1352.27 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-03-11 09:55:00 | 1393.40 | 2026-03-11 10:05:00 | 1399.80 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-03-11 09:55:00 | 1393.40 | 2026-03-11 15:20:00 | 1402.00 | TARGET_HIT | 0.50 | 0.62% |
| BUY | retest1 | 2026-04-10 11:10:00 | 1411.50 | 2026-04-10 11:45:00 | 1407.74 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-04-15 09:50:00 | 1406.90 | 2026-04-15 10:00:00 | 1416.13 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-04-15 09:50:00 | 1406.90 | 2026-04-15 13:55:00 | 1436.10 | TARGET_HIT | 0.50 | 2.08% |
| BUY | retest1 | 2026-04-21 09:55:00 | 1423.60 | 2026-04-21 11:05:00 | 1429.52 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-04-21 09:55:00 | 1423.60 | 2026-04-21 11:55:00 | 1423.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-23 10:40:00 | 1461.50 | 2026-04-23 11:00:00 | 1468.56 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-04-23 10:40:00 | 1461.50 | 2026-04-23 11:10:00 | 1461.50 | STOP_HIT | 0.50 | 0.00% |
