# ICICIBANK (ICICIBANK)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1267.80
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
| ENTRY1 | 17 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 4 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 13
- **Target hits / Stop hits / Partials:** 4 / 13 / 5
- **Avg / median % per leg:** 0.10% / -0.12%
- **Sum % (uncompounded):** 2.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 1 | 5 | 1 | -0.05% | -0.3% |
| BUY @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 1 | 5 | 1 | -0.05% | -0.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 15 | 7 | 46.7% | 3 | 8 | 4 | 0.17% | 2.6% |
| SELL @ 2nd Alert (retest1) | 15 | 7 | 46.7% | 3 | 8 | 4 | 0.17% | 2.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 9 | 40.9% | 4 | 13 | 5 | 0.10% | 2.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 09:30:00 | 1423.10 | 1418.47 | 0.00 | ORB-long ORB[1405.30,1422.70] vol=2.4x ATR=2.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 13:10:00 | 1427.31 | 1423.22 | 0.00 | T1 1.5R @ 1427.31 |
| Target hit | 2026-02-12 15:20:00 | 1430.10 | 1426.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:15:00 | 1399.20 | 1402.83 | 0.00 | ORB-short ORB[1401.30,1412.30] vol=1.6x ATR=1.64 |
| Stop hit — per-position SL triggered | 2026-02-18 11:20:00 | 1400.84 | 1402.36 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:35:00 | 1399.90 | 1401.33 | 0.00 | ORB-short ORB[1403.50,1412.30] vol=13.1x ATR=1.82 |
| Stop hit — per-position SL triggered | 2026-02-19 10:40:00 | 1401.72 | 1401.31 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 11:10:00 | 1398.70 | 1396.17 | 0.00 | ORB-long ORB[1384.00,1395.70] vol=1.9x ATR=2.38 |
| Stop hit — per-position SL triggered | 2026-02-20 12:40:00 | 1396.32 | 1396.78 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:35:00 | 1399.80 | 1396.30 | 0.00 | ORB-long ORB[1388.20,1398.10] vol=1.9x ATR=3.06 |
| Stop hit — per-position SL triggered | 2026-02-25 10:25:00 | 1396.74 | 1398.00 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 11:05:00 | 1390.50 | 1394.12 | 0.00 | ORB-short ORB[1394.30,1402.20] vol=1.8x ATR=2.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 12:35:00 | 1386.97 | 1392.71 | 0.00 | T1 1.5R @ 1386.97 |
| Target hit | 2026-02-27 15:20:00 | 1377.90 | 1386.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2026-03-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 11:05:00 | 1369.70 | 1361.41 | 0.00 | ORB-long ORB[1355.00,1368.90] vol=3.1x ATR=2.97 |
| Stop hit — per-position SL triggered | 2026-03-05 11:20:00 | 1366.73 | 1361.93 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:55:00 | 1226.70 | 1236.43 | 0.00 | ORB-short ORB[1231.60,1241.30] vol=1.6x ATR=3.65 |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 1230.35 | 1235.46 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 09:30:00 | 1220.70 | 1224.42 | 0.00 | ORB-short ORB[1221.20,1234.00] vol=2.6x ATR=4.67 |
| Stop hit — per-position SL triggered | 2026-04-01 09:45:00 | 1225.37 | 1224.13 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 11:15:00 | 1341.40 | 1350.29 | 0.00 | ORB-short ORB[1351.10,1364.80] vol=1.5x ATR=3.16 |
| Stop hit — per-position SL triggered | 2026-04-15 12:00:00 | 1344.56 | 1348.98 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-20 09:30:00 | 1373.60 | 1367.20 | 0.00 | ORB-long ORB[1355.10,1371.90] vol=3.0x ATR=4.12 |
| Stop hit — per-position SL triggered | 2026-04-20 09:40:00 | 1369.48 | 1368.05 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:40:00 | 1380.50 | 1372.60 | 0.00 | ORB-long ORB[1360.00,1379.80] vol=2.8x ATR=2.74 |
| Stop hit — per-position SL triggered | 2026-04-21 10:45:00 | 1377.76 | 1372.77 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:50:00 | 1334.60 | 1336.06 | 0.00 | ORB-short ORB[1335.00,1348.80] vol=2.1x ATR=2.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 13:10:00 | 1330.85 | 1334.60 | 0.00 | T1 1.5R @ 1330.85 |
| Target hit | 2026-04-24 15:20:00 | 1327.20 | 1332.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2026-04-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 10:35:00 | 1313.50 | 1326.89 | 0.00 | ORB-short ORB[1326.10,1335.90] vol=1.7x ATR=2.86 |
| Stop hit — per-position SL triggered | 2026-04-27 10:40:00 | 1316.36 | 1324.79 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:05:00 | 1310.20 | 1312.85 | 0.00 | ORB-short ORB[1312.00,1317.70] vol=1.6x ATR=2.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:20:00 | 1307.00 | 1312.55 | 0.00 | T1 1.5R @ 1307.00 |
| Target hit | 2026-04-28 15:20:00 | 1288.90 | 1300.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2026-05-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:10:00 | 1250.80 | 1255.10 | 0.00 | ORB-short ORB[1255.10,1266.70] vol=2.8x ATR=2.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:40:00 | 1246.66 | 1252.62 | 0.00 | T1 1.5R @ 1246.66 |
| Stop hit — per-position SL triggered | 2026-05-05 12:20:00 | 1250.80 | 1251.72 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-05-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:55:00 | 1264.70 | 1270.42 | 0.00 | ORB-short ORB[1266.80,1279.60] vol=2.1x ATR=2.46 |
| Stop hit — per-position SL triggered | 2026-05-08 11:05:00 | 1267.16 | 1270.20 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-12 09:30:00 | 1423.10 | 2026-02-12 13:10:00 | 1427.31 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-02-12 09:30:00 | 1423.10 | 2026-02-12 15:20:00 | 1430.10 | TARGET_HIT | 0.50 | 0.49% |
| SELL | retest1 | 2026-02-18 11:15:00 | 1399.20 | 2026-02-18 11:20:00 | 1400.84 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest1 | 2026-02-19 10:35:00 | 1399.90 | 2026-02-19 10:40:00 | 1401.72 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2026-02-20 11:10:00 | 1398.70 | 2026-02-20 12:40:00 | 1396.32 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2026-02-25 09:35:00 | 1399.80 | 2026-02-25 10:25:00 | 1396.74 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-27 11:05:00 | 1390.50 | 2026-02-27 12:35:00 | 1386.97 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2026-02-27 11:05:00 | 1390.50 | 2026-02-27 15:20:00 | 1377.90 | TARGET_HIT | 0.50 | 0.91% |
| BUY | retest1 | 2026-03-05 11:05:00 | 1369.70 | 2026-03-05 11:20:00 | 1366.73 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-03-24 10:55:00 | 1226.70 | 2026-03-24 11:15:00 | 1230.35 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-04-01 09:30:00 | 1220.70 | 2026-04-01 09:45:00 | 1225.37 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-04-15 11:15:00 | 1341.40 | 2026-04-15 12:00:00 | 1344.56 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-04-20 09:30:00 | 1373.60 | 2026-04-20 09:40:00 | 1369.48 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-21 10:40:00 | 1380.50 | 2026-04-21 10:45:00 | 1377.76 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-04-24 10:50:00 | 1334.60 | 2026-04-24 13:10:00 | 1330.85 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2026-04-24 10:50:00 | 1334.60 | 2026-04-24 15:20:00 | 1327.20 | TARGET_HIT | 0.50 | 0.55% |
| SELL | retest1 | 2026-04-27 10:35:00 | 1313.50 | 2026-04-27 10:40:00 | 1316.36 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-04-28 11:05:00 | 1310.20 | 2026-04-28 11:20:00 | 1307.00 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2026-04-28 11:05:00 | 1310.20 | 2026-04-28 15:20:00 | 1288.90 | TARGET_HIT | 0.50 | 1.63% |
| SELL | retest1 | 2026-05-05 10:10:00 | 1250.80 | 2026-05-05 11:40:00 | 1246.66 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-05-05 10:10:00 | 1250.80 | 2026-05-05 12:20:00 | 1250.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-08 10:55:00 | 1264.70 | 2026-05-08 11:05:00 | 1267.16 | STOP_HIT | 1.00 | -0.19% |
