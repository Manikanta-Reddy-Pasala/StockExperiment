# Affle 3i Ltd. (AFFLE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1510.00
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
| TARGET_HIT | 4 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 9
- **Target hits / Stop hits / Partials:** 4 / 9 / 6
- **Avg / median % per leg:** 0.20% / 0.42%
- **Sum % (uncompounded):** 3.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 6 | 85.7% | 3 | 1 | 3 | 0.48% | 3.4% |
| BUY @ 2nd Alert (retest1) | 7 | 6 | 85.7% | 3 | 1 | 3 | 0.48% | 3.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 4 | 33.3% | 1 | 8 | 3 | 0.03% | 0.4% |
| SELL @ 2nd Alert (retest1) | 12 | 4 | 33.3% | 1 | 8 | 3 | 0.03% | 0.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 10 | 52.6% | 4 | 9 | 6 | 0.20% | 3.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 1590.20 | 1598.63 | 0.00 | ORB-short ORB[1595.00,1610.00] vol=3.4x ATR=5.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 09:40:00 | 1582.37 | 1593.53 | 0.00 | T1 1.5R @ 1582.37 |
| Target hit | 2026-02-13 10:30:00 | 1581.80 | 1580.24 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — SELL (started 2026-02-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:45:00 | 1527.00 | 1540.21 | 0.00 | ORB-short ORB[1549.80,1567.70] vol=5.8x ATR=4.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:25:00 | 1519.81 | 1536.33 | 0.00 | T1 1.5R @ 1519.81 |
| Stop hit — per-position SL triggered | 2026-02-18 13:20:00 | 1527.00 | 1529.76 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 11:10:00 | 1519.50 | 1515.09 | 0.00 | ORB-long ORB[1499.00,1517.80] vol=2.6x ATR=4.72 |
| Stop hit — per-position SL triggered | 2026-02-23 11:20:00 | 1514.78 | 1515.20 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 11:05:00 | 1371.10 | 1390.96 | 0.00 | ORB-short ORB[1400.00,1416.50] vol=2.7x ATR=5.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:20:00 | 1363.23 | 1388.02 | 0.00 | T1 1.5R @ 1363.23 |
| Stop hit — per-position SL triggered | 2026-02-25 11:55:00 | 1371.10 | 1384.52 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 09:35:00 | 1340.90 | 1334.18 | 0.00 | ORB-long ORB[1321.10,1338.40] vol=2.7x ATR=5.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:40:00 | 1348.60 | 1336.71 | 0.00 | T1 1.5R @ 1348.60 |
| Target hit | 2026-02-27 10:05:00 | 1347.90 | 1347.92 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — SELL (started 2026-04-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 10:20:00 | 1400.00 | 1405.98 | 0.00 | ORB-short ORB[1402.10,1416.70] vol=2.3x ATR=4.30 |
| Stop hit — per-position SL triggered | 2026-04-10 14:05:00 | 1404.30 | 1403.84 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 10:50:00 | 1413.50 | 1423.71 | 0.00 | ORB-short ORB[1422.30,1440.00] vol=1.6x ATR=4.21 |
| Stop hit — per-position SL triggered | 2026-04-15 11:10:00 | 1417.71 | 1423.35 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:40:00 | 1462.90 | 1452.28 | 0.00 | ORB-long ORB[1442.00,1452.00] vol=2.4x ATR=4.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:45:00 | 1470.11 | 1460.33 | 0.00 | T1 1.5R @ 1470.11 |
| Target hit | 2026-04-21 11:00:00 | 1469.00 | 1469.99 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — SELL (started 2026-04-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:10:00 | 1434.40 | 1440.79 | 0.00 | ORB-short ORB[1438.60,1452.90] vol=5.3x ATR=3.60 |
| Stop hit — per-position SL triggered | 2026-04-23 11:30:00 | 1438.00 | 1440.63 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:40:00 | 1428.70 | 1422.84 | 0.00 | ORB-long ORB[1406.80,1426.20] vol=1.7x ATR=5.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:45:00 | 1437.27 | 1425.31 | 0.00 | T1 1.5R @ 1437.27 |
| Target hit | 2026-04-27 14:05:00 | 1444.00 | 1444.53 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — SELL (started 2026-04-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:00:00 | 1430.70 | 1435.72 | 0.00 | ORB-short ORB[1431.20,1448.00] vol=1.9x ATR=4.73 |
| Stop hit — per-position SL triggered | 2026-04-29 10:20:00 | 1435.43 | 1434.81 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-05-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 10:30:00 | 1414.10 | 1422.86 | 0.00 | ORB-short ORB[1418.10,1437.70] vol=1.6x ATR=4.02 |
| Stop hit — per-position SL triggered | 2026-05-04 10:40:00 | 1418.12 | 1422.52 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-05-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:05:00 | 1419.40 | 1426.74 | 0.00 | ORB-short ORB[1424.40,1434.60] vol=3.0x ATR=3.29 |
| Stop hit — per-position SL triggered | 2026-05-06 11:50:00 | 1422.69 | 1425.37 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-13 09:30:00 | 1590.20 | 2026-02-13 09:40:00 | 1582.37 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-02-13 09:30:00 | 1590.20 | 2026-02-13 10:30:00 | 1581.80 | TARGET_HIT | 0.50 | 0.53% |
| SELL | retest1 | 2026-02-18 10:45:00 | 1527.00 | 2026-02-18 11:25:00 | 1519.81 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-02-18 10:45:00 | 1527.00 | 2026-02-18 13:20:00 | 1527.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-23 11:10:00 | 1519.50 | 2026-02-23 11:20:00 | 1514.78 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-25 11:05:00 | 1371.10 | 2026-02-25 11:20:00 | 1363.23 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-02-25 11:05:00 | 1371.10 | 2026-02-25 11:55:00 | 1371.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-27 09:35:00 | 1340.90 | 2026-02-27 09:40:00 | 1348.60 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-02-27 09:35:00 | 1340.90 | 2026-02-27 10:05:00 | 1347.90 | TARGET_HIT | 0.50 | 0.52% |
| SELL | retest1 | 2026-04-10 10:20:00 | 1400.00 | 2026-04-10 14:05:00 | 1404.30 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-04-15 10:50:00 | 1413.50 | 2026-04-15 11:10:00 | 1417.71 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-21 09:40:00 | 1462.90 | 2026-04-21 09:45:00 | 1470.11 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-04-21 09:40:00 | 1462.90 | 2026-04-21 11:00:00 | 1469.00 | TARGET_HIT | 0.50 | 0.42% |
| SELL | retest1 | 2026-04-23 11:10:00 | 1434.40 | 2026-04-23 11:30:00 | 1438.00 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-04-27 09:40:00 | 1428.70 | 2026-04-27 09:45:00 | 1437.27 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-04-27 09:40:00 | 1428.70 | 2026-04-27 14:05:00 | 1444.00 | TARGET_HIT | 0.50 | 1.07% |
| SELL | retest1 | 2026-04-29 10:00:00 | 1430.70 | 2026-04-29 10:20:00 | 1435.43 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-05-04 10:30:00 | 1414.10 | 2026-05-04 10:40:00 | 1418.12 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-05-06 11:05:00 | 1419.40 | 2026-05-06 11:50:00 | 1422.69 | STOP_HIT | 1.00 | -0.23% |
