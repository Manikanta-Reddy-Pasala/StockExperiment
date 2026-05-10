# Eris Lifesciences Ltd. (ERIS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1389.70
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
| PARTIAL | 5 |
| TARGET_HIT | 4 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 11
- **Target hits / Stop hits / Partials:** 4 / 11 / 5
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 3.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 6 | 42.9% | 3 | 8 | 3 | 0.22% | 3.1% |
| BUY @ 2nd Alert (retest1) | 14 | 6 | 42.9% | 3 | 8 | 3 | 0.22% | 3.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.09% | 0.5% |
| SELL @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.09% | 0.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 20 | 9 | 45.0% | 4 | 11 | 5 | 0.18% | 3.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:10:00 | 1445.40 | 1439.46 | 0.00 | ORB-long ORB[1427.30,1445.20] vol=2.7x ATR=7.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 12:15:00 | 1455.90 | 1444.70 | 0.00 | T1 1.5R @ 1455.90 |
| Target hit | 2026-02-09 15:20:00 | 1472.20 | 1461.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2026-02-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:00:00 | 1492.00 | 1476.00 | 0.00 | ORB-long ORB[1462.00,1479.90] vol=1.7x ATR=5.65 |
| Stop hit — per-position SL triggered | 2026-02-12 10:05:00 | 1486.35 | 1476.42 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 11:10:00 | 1348.60 | 1342.23 | 0.00 | ORB-long ORB[1335.00,1345.00] vol=2.6x ATR=3.42 |
| Stop hit — per-position SL triggered | 2026-02-19 12:30:00 | 1345.18 | 1344.01 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 11:05:00 | 1358.80 | 1355.17 | 0.00 | ORB-long ORB[1347.30,1358.50] vol=4.2x ATR=2.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 12:10:00 | 1363.29 | 1371.38 | 0.00 | T1 1.5R @ 1363.29 |
| Target hit | 2026-02-24 12:15:00 | 1374.10 | 1375.79 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — BUY (started 2026-02-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:55:00 | 1383.20 | 1379.35 | 0.00 | ORB-long ORB[1370.10,1383.00] vol=3.7x ATR=2.70 |
| Stop hit — per-position SL triggered | 2026-02-26 10:25:00 | 1380.50 | 1380.30 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 10:50:00 | 1368.90 | 1364.31 | 0.00 | ORB-long ORB[1361.00,1368.40] vol=5.5x ATR=2.78 |
| Stop hit — per-position SL triggered | 2026-03-06 10:55:00 | 1366.12 | 1364.37 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-12 10:50:00 | 1331.60 | 1337.58 | 0.00 | ORB-short ORB[1332.10,1350.10] vol=2.2x ATR=3.46 |
| Stop hit — per-position SL triggered | 2026-03-12 11:00:00 | 1335.06 | 1337.43 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:20:00 | 1339.60 | 1343.96 | 0.00 | ORB-short ORB[1341.50,1351.90] vol=1.9x ATR=4.75 |
| Stop hit — per-position SL triggered | 2026-03-13 10:25:00 | 1344.35 | 1343.94 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:35:00 | 1291.90 | 1285.54 | 0.00 | ORB-long ORB[1272.60,1290.10] vol=2.4x ATR=5.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 10:25:00 | 1299.50 | 1290.61 | 0.00 | T1 1.5R @ 1299.50 |
| Target hit | 2026-03-18 15:20:00 | 1304.40 | 1300.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2026-04-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 10:55:00 | 1448.70 | 1458.66 | 0.00 | ORB-short ORB[1452.00,1472.90] vol=3.5x ATR=3.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 14:30:00 | 1443.17 | 1448.15 | 0.00 | T1 1.5R @ 1443.17 |
| Target hit | 2026-04-17 15:20:00 | 1446.10 | 1447.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2026-04-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:55:00 | 1382.00 | 1400.57 | 0.00 | ORB-short ORB[1399.30,1416.30] vol=1.6x ATR=5.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:45:00 | 1373.83 | 1398.17 | 0.00 | T1 1.5R @ 1373.83 |
| Stop hit — per-position SL triggered | 2026-04-24 15:15:00 | 1382.00 | 1381.41 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:35:00 | 1358.60 | 1347.33 | 0.00 | ORB-long ORB[1331.00,1344.90] vol=2.0x ATR=6.18 |
| Stop hit — per-position SL triggered | 2026-04-29 09:45:00 | 1352.42 | 1349.35 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-05-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 11:00:00 | 1379.90 | 1367.82 | 0.00 | ORB-long ORB[1358.10,1367.00] vol=2.3x ATR=4.23 |
| Stop hit — per-position SL triggered | 2026-05-06 11:10:00 | 1375.67 | 1368.39 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:05:00 | 1388.00 | 1384.11 | 0.00 | ORB-long ORB[1375.10,1384.90] vol=6.4x ATR=4.84 |
| Stop hit — per-position SL triggered | 2026-05-07 10:45:00 | 1383.16 | 1384.61 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-05-08 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:20:00 | 1404.20 | 1396.77 | 0.00 | ORB-long ORB[1382.00,1402.10] vol=1.6x ATR=5.52 |
| Stop hit — per-position SL triggered | 2026-05-08 11:55:00 | 1398.68 | 1399.34 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 11:10:00 | 1445.40 | 2026-02-09 12:15:00 | 1455.90 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2026-02-09 11:10:00 | 1445.40 | 2026-02-09 15:20:00 | 1472.20 | TARGET_HIT | 0.50 | 1.85% |
| BUY | retest1 | 2026-02-12 10:00:00 | 1492.00 | 2026-02-12 10:05:00 | 1486.35 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-02-19 11:10:00 | 1348.60 | 2026-02-19 12:30:00 | 1345.18 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-24 11:05:00 | 1358.80 | 2026-02-24 12:10:00 | 1363.29 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-02-24 11:05:00 | 1358.80 | 2026-02-24 12:15:00 | 1374.10 | TARGET_HIT | 0.50 | 1.13% |
| BUY | retest1 | 2026-02-26 09:55:00 | 1383.20 | 2026-02-26 10:25:00 | 1380.50 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-03-06 10:50:00 | 1368.90 | 2026-03-06 10:55:00 | 1366.12 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-03-12 10:50:00 | 1331.60 | 2026-03-12 11:00:00 | 1335.06 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-03-13 10:20:00 | 1339.60 | 2026-03-13 10:25:00 | 1344.35 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-03-18 09:35:00 | 1291.90 | 2026-03-18 10:25:00 | 1299.50 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2026-03-18 09:35:00 | 1291.90 | 2026-03-18 15:20:00 | 1304.40 | TARGET_HIT | 0.50 | 0.97% |
| SELL | retest1 | 2026-04-17 10:55:00 | 1448.70 | 2026-04-17 14:30:00 | 1443.17 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-04-17 10:55:00 | 1448.70 | 2026-04-17 15:20:00 | 1446.10 | TARGET_HIT | 0.50 | 0.18% |
| SELL | retest1 | 2026-04-24 10:55:00 | 1382.00 | 2026-04-24 11:45:00 | 1373.83 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-04-24 10:55:00 | 1382.00 | 2026-04-24 15:15:00 | 1382.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 09:35:00 | 1358.60 | 2026-04-29 09:45:00 | 1352.42 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-05-06 11:00:00 | 1379.90 | 2026-05-06 11:10:00 | 1375.67 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-05-07 10:05:00 | 1388.00 | 2026-05-07 10:45:00 | 1383.16 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-05-08 10:20:00 | 1404.20 | 2026-05-08 11:55:00 | 1398.68 | STOP_HIT | 1.00 | -0.39% |
