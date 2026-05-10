# Aavas Financiers Ltd. (AAVAS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1446.00
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
- **Avg / median % per leg:** 0.34% / 0.30%
- **Sum % (uncompounded):** 6.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 6 | 66.7% | 3 | 3 | 3 | 0.60% | 5.4% |
| BUY @ 2nd Alert (retest1) | 9 | 6 | 66.7% | 3 | 3 | 3 | 0.60% | 5.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.11% | 1.1% |
| SELL @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.11% | 1.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 10 | 52.6% | 4 | 9 | 6 | 0.34% | 6.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 11:00:00 | 1293.80 | 1280.74 | 0.00 | ORB-long ORB[1266.60,1281.70] vol=3.6x ATR=3.48 |
| Stop hit — per-position SL triggered | 2026-02-16 11:25:00 | 1290.32 | 1282.75 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:40:00 | 1304.10 | 1310.12 | 0.00 | ORB-short ORB[1304.60,1322.90] vol=1.6x ATR=4.38 |
| Stop hit — per-position SL triggered | 2026-02-18 10:30:00 | 1308.48 | 1307.65 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:35:00 | 1299.70 | 1301.99 | 0.00 | ORB-short ORB[1301.40,1311.00] vol=2.3x ATR=2.61 |
| Stop hit — per-position SL triggered | 2026-02-19 10:50:00 | 1302.31 | 1301.71 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:40:00 | 1258.10 | 1265.27 | 0.00 | ORB-short ORB[1260.00,1273.20] vol=1.6x ATR=3.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 09:45:00 | 1252.93 | 1257.78 | 0.00 | T1 1.5R @ 1252.93 |
| Stop hit — per-position SL triggered | 2026-02-25 10:00:00 | 1258.10 | 1257.36 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:25:00 | 1221.00 | 1228.52 | 0.00 | ORB-short ORB[1225.00,1238.40] vol=1.9x ATR=4.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:30:00 | 1214.23 | 1227.19 | 0.00 | T1 1.5R @ 1214.23 |
| Stop hit — per-position SL triggered | 2026-03-05 12:25:00 | 1221.00 | 1221.46 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 1220.00 | 1227.27 | 0.00 | ORB-short ORB[1222.30,1233.00] vol=4.0x ATR=3.37 |
| Stop hit — per-position SL triggered | 2026-03-06 11:00:00 | 1223.37 | 1226.93 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:10:00 | 1337.00 | 1325.98 | 0.00 | ORB-long ORB[1313.80,1324.70] vol=2.6x ATR=3.58 |
| Stop hit — per-position SL triggered | 2026-04-17 10:35:00 | 1333.42 | 1328.52 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-20 10:10:00 | 1349.00 | 1337.34 | 0.00 | ORB-long ORB[1330.00,1345.60] vol=1.8x ATR=4.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-20 11:15:00 | 1356.41 | 1341.10 | 0.00 | T1 1.5R @ 1356.41 |
| Target hit | 2026-04-20 15:15:00 | 1353.10 | 1354.91 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2026-04-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:50:00 | 1388.50 | 1374.86 | 0.00 | ORB-long ORB[1355.80,1375.00] vol=5.1x ATR=5.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:15:00 | 1396.91 | 1378.59 | 0.00 | T1 1.5R @ 1396.91 |
| Target hit | 2026-04-21 11:55:00 | 1396.90 | 1398.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — SELL (started 2026-04-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 10:00:00 | 1395.60 | 1404.62 | 0.00 | ORB-short ORB[1401.00,1418.60] vol=1.7x ATR=4.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 10:25:00 | 1389.24 | 1398.38 | 0.00 | T1 1.5R @ 1389.24 |
| Target hit | 2026-04-27 15:20:00 | 1383.00 | 1389.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2026-05-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 11:00:00 | 1388.40 | 1386.59 | 0.00 | ORB-long ORB[1374.70,1386.80] vol=16.5x ATR=3.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:25:00 | 1394.07 | 1386.98 | 0.00 | T1 1.5R @ 1394.07 |
| Target hit | 2026-05-05 15:20:00 | 1441.40 | 1418.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2026-05-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 10:45:00 | 1413.00 | 1423.98 | 0.00 | ORB-short ORB[1430.40,1446.80] vol=7.0x ATR=5.99 |
| Stop hit — per-position SL triggered | 2026-05-07 11:50:00 | 1418.99 | 1421.59 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-05-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 11:00:00 | 1446.30 | 1440.49 | 0.00 | ORB-long ORB[1415.00,1436.60] vol=1.5x ATR=5.65 |
| Stop hit — per-position SL triggered | 2026-05-08 12:00:00 | 1440.65 | 1441.43 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-16 11:00:00 | 1293.80 | 2026-02-16 11:25:00 | 1290.32 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-02-18 09:40:00 | 1304.10 | 2026-02-18 10:30:00 | 1308.48 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-02-19 10:35:00 | 1299.70 | 2026-02-19 10:50:00 | 1302.31 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-25 09:40:00 | 1258.10 | 2026-02-25 09:45:00 | 1252.93 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-02-25 09:40:00 | 1258.10 | 2026-02-25 10:00:00 | 1258.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-05 10:25:00 | 1221.00 | 2026-03-05 10:30:00 | 1214.23 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-03-05 10:25:00 | 1221.00 | 2026-03-05 12:25:00 | 1221.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-06 10:45:00 | 1220.00 | 2026-03-06 11:00:00 | 1223.37 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-17 10:10:00 | 1337.00 | 2026-04-17 10:35:00 | 1333.42 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-04-20 10:10:00 | 1349.00 | 2026-04-20 11:15:00 | 1356.41 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-04-20 10:10:00 | 1349.00 | 2026-04-20 15:15:00 | 1353.10 | TARGET_HIT | 0.50 | 0.30% |
| BUY | retest1 | 2026-04-21 09:50:00 | 1388.50 | 2026-04-21 10:15:00 | 1396.91 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-04-21 09:50:00 | 1388.50 | 2026-04-21 11:55:00 | 1396.90 | TARGET_HIT | 0.50 | 0.60% |
| SELL | retest1 | 2026-04-27 10:00:00 | 1395.60 | 2026-04-27 10:25:00 | 1389.24 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-04-27 10:00:00 | 1395.60 | 2026-04-27 15:20:00 | 1383.00 | TARGET_HIT | 0.50 | 0.90% |
| BUY | retest1 | 2026-05-05 11:00:00 | 1388.40 | 2026-05-05 11:25:00 | 1394.07 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-05-05 11:00:00 | 1388.40 | 2026-05-05 15:20:00 | 1441.40 | TARGET_HIT | 0.50 | 3.82% |
| SELL | retest1 | 2026-05-07 10:45:00 | 1413.00 | 2026-05-07 11:50:00 | 1418.99 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-05-08 11:00:00 | 1446.30 | 2026-05-08 12:00:00 | 1440.65 | STOP_HIT | 1.00 | -0.39% |
