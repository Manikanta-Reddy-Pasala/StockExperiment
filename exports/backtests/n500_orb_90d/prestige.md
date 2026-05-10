# Prestige Estates Projects Ltd. (PRESTIGE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1495.50
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
| PARTIAL | 8 |
| TARGET_HIT | 4 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 9
- **Target hits / Stop hits / Partials:** 4 / 9 / 8
- **Avg / median % per leg:** 0.46% / 0.52%
- **Sum % (uncompounded):** 9.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 5 | 62.5% | 2 | 3 | 3 | 0.59% | 4.7% |
| BUY @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 2 | 3 | 3 | 0.59% | 4.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 13 | 7 | 53.8% | 2 | 6 | 5 | 0.38% | 4.9% |
| SELL @ 2nd Alert (retest1) | 13 | 7 | 53.8% | 2 | 6 | 5 | 0.38% | 4.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 21 | 12 | 57.1% | 4 | 9 | 8 | 0.46% | 9.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:40:00 | 1509.30 | 1515.98 | 0.00 | ORB-short ORB[1522.10,1538.40] vol=4.5x ATR=4.56 |
| Stop hit — per-position SL triggered | 2026-02-18 13:50:00 | 1513.86 | 1510.08 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:35:00 | 1503.10 | 1517.89 | 0.00 | ORB-short ORB[1518.00,1538.00] vol=1.8x ATR=4.34 |
| Stop hit — per-position SL triggered | 2026-02-19 10:40:00 | 1507.44 | 1517.15 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:45:00 | 1464.50 | 1469.27 | 0.00 | ORB-short ORB[1466.10,1483.10] vol=2.6x ATR=5.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:35:00 | 1456.73 | 1465.33 | 0.00 | T1 1.5R @ 1456.73 |
| Target hit | 2026-02-24 15:20:00 | 1439.00 | 1431.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 1354.30 | 1366.64 | 0.00 | ORB-short ORB[1357.50,1377.40] vol=2.1x ATR=4.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 11:00:00 | 1347.61 | 1364.48 | 0.00 | T1 1.5R @ 1347.61 |
| Stop hit — per-position SL triggered | 2026-03-06 11:35:00 | 1354.30 | 1358.43 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 10:35:00 | 1242.20 | 1249.23 | 0.00 | ORB-short ORB[1246.20,1263.70] vol=1.7x ATR=5.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 11:45:00 | 1233.42 | 1244.51 | 0.00 | T1 1.5R @ 1233.42 |
| Stop hit — per-position SL triggered | 2026-03-17 12:45:00 | 1242.20 | 1242.24 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 10:40:00 | 1280.90 | 1272.74 | 0.00 | ORB-long ORB[1257.30,1273.90] vol=1.7x ATR=5.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 10:45:00 | 1288.65 | 1275.54 | 0.00 | T1 1.5R @ 1288.65 |
| Target hit | 2026-03-18 15:20:00 | 1320.60 | 1300.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2026-03-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:30:00 | 1300.20 | 1291.01 | 0.00 | ORB-long ORB[1278.90,1291.90] vol=1.7x ATR=6.31 |
| Stop hit — per-position SL triggered | 2026-03-20 09:55:00 | 1293.89 | 1293.74 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:45:00 | 1304.00 | 1310.44 | 0.00 | ORB-short ORB[1309.60,1320.80] vol=1.8x ATR=5.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 09:55:00 | 1295.64 | 1308.06 | 0.00 | T1 1.5R @ 1295.64 |
| Stop hit — per-position SL triggered | 2026-04-09 10:05:00 | 1304.00 | 1307.22 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:00:00 | 1355.00 | 1345.36 | 0.00 | ORB-long ORB[1334.40,1351.20] vol=1.9x ATR=5.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 14:45:00 | 1363.86 | 1353.84 | 0.00 | T1 1.5R @ 1363.86 |
| Stop hit — per-position SL triggered | 2026-04-17 15:00:00 | 1355.00 | 1353.99 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 1382.10 | 1388.72 | 0.00 | ORB-short ORB[1384.20,1399.40] vol=1.7x ATR=6.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:30:00 | 1373.08 | 1383.30 | 0.00 | T1 1.5R @ 1373.08 |
| Target hit | 2026-04-24 14:15:00 | 1367.00 | 1366.75 | 0.00 | Trail-exit close>VWAP |

### Cycle 11 — BUY (started 2026-04-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:55:00 | 1425.00 | 1415.27 | 0.00 | ORB-long ORB[1406.10,1421.80] vol=1.6x ATR=4.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:00:00 | 1432.38 | 1419.57 | 0.00 | T1 1.5R @ 1432.38 |
| Target hit | 2026-04-29 14:15:00 | 1435.00 | 1436.42 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — SELL (started 2026-04-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:25:00 | 1399.40 | 1412.25 | 0.00 | ORB-short ORB[1409.30,1430.00] vol=1.6x ATR=5.11 |
| Stop hit — per-position SL triggered | 2026-04-30 10:40:00 | 1404.51 | 1411.64 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-05-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:35:00 | 1501.60 | 1490.80 | 0.00 | ORB-long ORB[1480.00,1496.10] vol=2.2x ATR=5.37 |
| Stop hit — per-position SL triggered | 2026-05-08 11:00:00 | 1496.23 | 1492.80 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-18 10:40:00 | 1509.30 | 2026-02-18 13:50:00 | 1513.86 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-19 10:35:00 | 1503.10 | 2026-02-19 10:40:00 | 1507.44 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-24 09:45:00 | 1464.50 | 2026-02-24 11:35:00 | 1456.73 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-02-24 09:45:00 | 1464.50 | 2026-02-24 15:20:00 | 1439.00 | TARGET_HIT | 0.50 | 1.74% |
| SELL | retest1 | 2026-03-06 10:45:00 | 1354.30 | 2026-03-06 11:00:00 | 1347.61 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-03-06 10:45:00 | 1354.30 | 2026-03-06 11:35:00 | 1354.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-17 10:35:00 | 1242.20 | 2026-03-17 11:45:00 | 1233.42 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2026-03-17 10:35:00 | 1242.20 | 2026-03-17 12:45:00 | 1242.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-18 10:40:00 | 1280.90 | 2026-03-18 10:45:00 | 1288.65 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-03-18 10:40:00 | 1280.90 | 2026-03-18 15:20:00 | 1320.60 | TARGET_HIT | 0.50 | 3.10% |
| BUY | retest1 | 2026-03-20 09:30:00 | 1300.20 | 2026-03-20 09:55:00 | 1293.89 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-04-09 09:45:00 | 1304.00 | 2026-04-09 09:55:00 | 1295.64 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2026-04-09 09:45:00 | 1304.00 | 2026-04-09 10:05:00 | 1304.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 10:00:00 | 1355.00 | 2026-04-17 14:45:00 | 1363.86 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-04-17 10:00:00 | 1355.00 | 2026-04-17 15:00:00 | 1355.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-24 09:30:00 | 1382.10 | 2026-04-24 10:30:00 | 1373.08 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2026-04-24 09:30:00 | 1382.10 | 2026-04-24 14:15:00 | 1367.00 | TARGET_HIT | 0.50 | 1.09% |
| BUY | retest1 | 2026-04-29 09:55:00 | 1425.00 | 2026-04-29 10:00:00 | 1432.38 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-04-29 09:55:00 | 1425.00 | 2026-04-29 14:15:00 | 1435.00 | TARGET_HIT | 0.50 | 0.70% |
| SELL | retest1 | 2026-04-30 10:25:00 | 1399.40 | 2026-04-30 10:40:00 | 1404.51 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-05-08 10:35:00 | 1501.60 | 2026-05-08 11:00:00 | 1496.23 | STOP_HIT | 1.00 | -0.36% |
