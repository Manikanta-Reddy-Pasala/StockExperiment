# Sobha Ltd. (SOBHA)

## Backtest Summary

- **Window:** 2026-03-09 09:15:00 → 2026-05-08 15:25:00 (3000 bars)
- **Last close:** 1425.00
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
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 12
- **Target hits / Stop hits / Partials:** 0 / 12 / 2
- **Avg / median % per leg:** -0.19% / -0.34%
- **Sum % (uncompounded):** -2.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.13% | -0.8% |
| BUY @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.13% | -0.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 1 | 12.5% | 0 | 7 | 1 | -0.24% | -1.9% |
| SELL @ 2nd Alert (retest1) | 8 | 1 | 12.5% | 0 | 7 | 1 | -0.24% | -1.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 14 | 2 | 14.3% | 0 | 12 | 2 | -0.19% | -2.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-03-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:45:00 | 1338.90 | 1311.40 | 0.00 | ORB-long ORB[1295.10,1310.00] vol=5.0x ATR=5.84 |
| Stop hit — per-position SL triggered | 2026-03-10 10:50:00 | 1333.06 | 1316.28 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-03-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 11:00:00 | 1342.20 | 1336.73 | 0.00 | ORB-long ORB[1321.00,1341.00] vol=3.9x ATR=3.46 |
| Stop hit — per-position SL triggered | 2026-03-11 13:15:00 | 1338.74 | 1338.48 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-03-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-24 10:45:00 | 1224.30 | 1211.76 | 0.00 | ORB-long ORB[1207.10,1221.10] vol=2.4x ATR=5.65 |
| Stop hit — per-position SL triggered | 2026-03-24 10:50:00 | 1218.65 | 1211.56 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-04-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:55:00 | 1276.70 | 1279.39 | 0.00 | ORB-short ORB[1282.50,1294.30] vol=3.1x ATR=6.28 |
| Stop hit — per-position SL triggered | 2026-04-09 10:15:00 | 1282.98 | 1279.27 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-04-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:40:00 | 1317.50 | 1327.98 | 0.00 | ORB-short ORB[1321.00,1339.90] vol=3.9x ATR=4.38 |
| Stop hit — per-position SL triggered | 2026-04-16 11:25:00 | 1321.88 | 1321.86 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:05:00 | 1333.80 | 1326.14 | 0.00 | ORB-long ORB[1313.40,1332.90] vol=2.0x ATR=5.06 |
| Stop hit — per-position SL triggered | 2026-04-17 10:10:00 | 1328.74 | 1326.79 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-20 09:35:00 | 1315.90 | 1325.06 | 0.00 | ORB-short ORB[1317.90,1337.40] vol=2.1x ATR=4.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-20 09:45:00 | 1308.59 | 1323.21 | 0.00 | T1 1.5R @ 1308.59 |
| Stop hit — per-position SL triggered | 2026-04-20 09:50:00 | 1315.90 | 1322.45 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:55:00 | 1355.80 | 1367.20 | 0.00 | ORB-short ORB[1365.70,1382.10] vol=1.8x ATR=5.70 |
| Stop hit — per-position SL triggered | 2026-04-22 11:05:00 | 1361.50 | 1366.16 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 10:50:00 | 1410.30 | 1415.75 | 0.00 | ORB-short ORB[1411.70,1430.30] vol=2.9x ATR=5.41 |
| Stop hit — per-position SL triggered | 2026-04-27 11:30:00 | 1415.71 | 1414.38 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:45:00 | 1416.50 | 1421.13 | 0.00 | ORB-short ORB[1419.50,1435.00] vol=2.8x ATR=4.80 |
| Stop hit — per-position SL triggered | 2026-04-28 10:20:00 | 1421.30 | 1419.91 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-29 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:30:00 | 1439.00 | 1452.63 | 0.00 | ORB-short ORB[1442.50,1464.10] vol=1.6x ATR=6.74 |
| Stop hit — per-position SL triggered | 2026-04-29 11:25:00 | 1445.74 | 1451.04 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-05-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:35:00 | 1462.80 | 1455.91 | 0.00 | ORB-long ORB[1439.90,1460.00] vol=2.3x ATR=7.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 09:55:00 | 1473.49 | 1462.85 | 0.00 | T1 1.5R @ 1473.49 |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 1462.80 | 1463.85 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-03-10 10:45:00 | 1338.90 | 2026-03-10 10:50:00 | 1333.06 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-03-11 11:00:00 | 1342.20 | 2026-03-11 13:15:00 | 1338.74 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-03-24 10:45:00 | 1224.30 | 2026-03-24 10:50:00 | 1218.65 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-04-09 09:55:00 | 1276.70 | 2026-04-09 10:15:00 | 1282.98 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-04-16 09:40:00 | 1317.50 | 2026-04-16 11:25:00 | 1321.88 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-17 10:05:00 | 1333.80 | 2026-04-17 10:10:00 | 1328.74 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-04-20 09:35:00 | 1315.90 | 2026-04-20 09:45:00 | 1308.59 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-04-20 09:35:00 | 1315.90 | 2026-04-20 09:50:00 | 1315.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-22 10:55:00 | 1355.80 | 2026-04-22 11:05:00 | 1361.50 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-04-27 10:50:00 | 1410.30 | 2026-04-27 11:30:00 | 1415.71 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-04-28 09:45:00 | 1416.50 | 2026-04-28 10:20:00 | 1421.30 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-04-29 10:30:00 | 1439.00 | 2026-04-29 11:25:00 | 1445.74 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-05-04 09:35:00 | 1462.80 | 2026-05-04 09:55:00 | 1473.49 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2026-05-04 09:35:00 | 1462.80 | 2026-05-04 10:15:00 | 1462.80 | STOP_HIT | 0.50 | 0.00% |
