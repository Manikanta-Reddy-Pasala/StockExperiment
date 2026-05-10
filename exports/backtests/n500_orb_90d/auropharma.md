# Aurobindo Pharma Ltd. (AUROPHARMA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1487.70
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
| PARTIAL | 11 |
| TARGET_HIT | 4 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 11
- **Target hits / Stop hits / Partials:** 4 / 11 / 11
- **Avg / median % per leg:** 0.50% / 0.42%
- **Sum % (uncompounded):** 13.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 10 | 66.7% | 3 | 5 | 7 | 0.66% | 9.9% |
| BUY @ 2nd Alert (retest1) | 15 | 10 | 66.7% | 3 | 5 | 7 | 0.66% | 9.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 5 | 45.5% | 1 | 6 | 4 | 0.29% | 3.2% |
| SELL @ 2nd Alert (retest1) | 11 | 5 | 45.5% | 1 | 6 | 4 | 0.29% | 3.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 26 | 15 | 57.7% | 4 | 11 | 11 | 0.50% | 13.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 1137.90 | 1147.17 | 0.00 | ORB-short ORB[1141.70,1154.30] vol=1.7x ATR=5.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 10:15:00 | 1130.23 | 1141.50 | 0.00 | T1 1.5R @ 1130.23 |
| Stop hit — per-position SL triggered | 2026-02-13 10:30:00 | 1137.90 | 1140.73 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:25:00 | 1142.90 | 1165.79 | 0.00 | ORB-short ORB[1179.10,1193.50] vol=2.1x ATR=7.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:30:00 | 1132.14 | 1157.33 | 0.00 | T1 1.5R @ 1132.14 |
| Stop hit — per-position SL triggered | 2026-02-18 10:35:00 | 1142.90 | 1155.08 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:35:00 | 1159.50 | 1154.24 | 0.00 | ORB-long ORB[1144.00,1156.70] vol=2.1x ATR=4.24 |
| Stop hit — per-position SL triggered | 2026-02-24 10:25:00 | 1155.26 | 1155.77 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:00:00 | 1180.10 | 1171.16 | 0.00 | ORB-long ORB[1162.40,1173.40] vol=2.0x ATR=3.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:25:00 | 1185.64 | 1175.95 | 0.00 | T1 1.5R @ 1185.64 |
| Target hit | 2026-02-25 15:20:00 | 1213.90 | 1204.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2026-03-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-12 09:45:00 | 1296.20 | 1303.74 | 0.00 | ORB-short ORB[1297.70,1314.00] vol=2.2x ATR=5.53 |
| Stop hit — per-position SL triggered | 2026-03-12 09:55:00 | 1301.73 | 1303.34 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 09:30:00 | 1296.70 | 1303.21 | 0.00 | ORB-short ORB[1297.00,1314.00] vol=1.8x ATR=6.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:55:00 | 1287.31 | 1300.09 | 0.00 | T1 1.5R @ 1287.31 |
| Target hit | 2026-03-16 15:20:00 | 1277.90 | 1285.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2026-03-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 10:50:00 | 1273.00 | 1280.57 | 0.00 | ORB-short ORB[1280.20,1299.00] vol=3.5x ATR=4.10 |
| Stop hit — per-position SL triggered | 2026-03-23 10:55:00 | 1277.10 | 1280.47 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:00:00 | 1302.20 | 1296.55 | 0.00 | ORB-long ORB[1282.10,1301.00] vol=2.6x ATR=5.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 10:10:00 | 1309.78 | 1297.92 | 0.00 | T1 1.5R @ 1309.78 |
| Target hit | 2026-03-25 13:25:00 | 1314.80 | 1314.91 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2026-03-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 09:40:00 | 1317.90 | 1309.70 | 0.00 | ORB-long ORB[1296.00,1314.40] vol=2.6x ATR=5.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 11:10:00 | 1325.80 | 1315.36 | 0.00 | T1 1.5R @ 1325.80 |
| Stop hit — per-position SL triggered | 2026-03-27 12:00:00 | 1317.90 | 1317.01 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 11:00:00 | 1402.90 | 1393.61 | 0.00 | ORB-long ORB[1382.40,1394.40] vol=2.0x ATR=3.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 11:15:00 | 1407.97 | 1397.13 | 0.00 | T1 1.5R @ 1407.97 |
| Stop hit — per-position SL triggered | 2026-04-22 11:40:00 | 1402.90 | 1401.02 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:40:00 | 1444.50 | 1430.15 | 0.00 | ORB-long ORB[1409.10,1430.20] vol=1.6x ATR=5.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 10:15:00 | 1452.86 | 1438.77 | 0.00 | T1 1.5R @ 1452.86 |
| Stop hit — per-position SL triggered | 2026-04-23 10:20:00 | 1444.50 | 1439.74 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:40:00 | 1428.00 | 1435.21 | 0.00 | ORB-short ORB[1430.10,1444.60] vol=1.8x ATR=3.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:35:00 | 1422.05 | 1433.45 | 0.00 | T1 1.5R @ 1422.05 |
| Stop hit — per-position SL triggered | 2026-04-24 12:45:00 | 1428.00 | 1430.70 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 09:55:00 | 1411.70 | 1417.47 | 0.00 | ORB-short ORB[1413.50,1433.00] vol=1.9x ATR=4.25 |
| Stop hit — per-position SL triggered | 2026-04-29 10:05:00 | 1415.95 | 1416.12 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 10:40:00 | 1387.30 | 1380.24 | 0.00 | ORB-long ORB[1367.00,1379.90] vol=1.7x ATR=3.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:50:00 | 1392.92 | 1381.24 | 0.00 | T1 1.5R @ 1392.92 |
| Target hit | 2026-05-05 15:20:00 | 1429.00 | 1405.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2026-05-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:00:00 | 1499.40 | 1491.53 | 0.00 | ORB-long ORB[1480.10,1498.00] vol=2.1x ATR=4.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 10:05:00 | 1506.59 | 1493.64 | 0.00 | T1 1.5R @ 1506.59 |
| Stop hit — per-position SL triggered | 2026-05-08 11:40:00 | 1499.40 | 1498.66 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-13 09:30:00 | 1137.90 | 2026-02-13 10:15:00 | 1130.23 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2026-02-13 09:30:00 | 1137.90 | 2026-02-13 10:30:00 | 1137.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-18 10:25:00 | 1142.90 | 2026-02-18 10:30:00 | 1132.14 | PARTIAL | 0.50 | 0.94% |
| SELL | retest1 | 2026-02-18 10:25:00 | 1142.90 | 2026-02-18 10:35:00 | 1142.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-24 09:35:00 | 1159.50 | 2026-02-24 10:25:00 | 1155.26 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-02-25 10:00:00 | 1180.10 | 2026-02-25 10:25:00 | 1185.64 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-02-25 10:00:00 | 1180.10 | 2026-02-25 15:20:00 | 1213.90 | TARGET_HIT | 0.50 | 2.86% |
| SELL | retest1 | 2026-03-12 09:45:00 | 1296.20 | 2026-03-12 09:55:00 | 1301.73 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-03-16 09:30:00 | 1296.70 | 2026-03-16 09:55:00 | 1287.31 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2026-03-16 09:30:00 | 1296.70 | 2026-03-16 15:20:00 | 1277.90 | TARGET_HIT | 0.50 | 1.45% |
| SELL | retest1 | 2026-03-23 10:50:00 | 1273.00 | 2026-03-23 10:55:00 | 1277.10 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-03-25 10:00:00 | 1302.20 | 2026-03-25 10:10:00 | 1309.78 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-03-25 10:00:00 | 1302.20 | 2026-03-25 13:25:00 | 1314.80 | TARGET_HIT | 0.50 | 0.97% |
| BUY | retest1 | 2026-03-27 09:40:00 | 1317.90 | 2026-03-27 11:10:00 | 1325.80 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-03-27 09:40:00 | 1317.90 | 2026-03-27 12:00:00 | 1317.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 11:00:00 | 1402.90 | 2026-04-22 11:15:00 | 1407.97 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-04-22 11:00:00 | 1402.90 | 2026-04-22 11:40:00 | 1402.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-23 09:40:00 | 1444.50 | 2026-04-23 10:15:00 | 1452.86 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-04-23 09:40:00 | 1444.50 | 2026-04-23 10:20:00 | 1444.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-24 10:40:00 | 1428.00 | 2026-04-24 11:35:00 | 1422.05 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-04-24 10:40:00 | 1428.00 | 2026-04-24 12:45:00 | 1428.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-29 09:55:00 | 1411.70 | 2026-04-29 10:05:00 | 1415.95 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-05 10:40:00 | 1387.30 | 2026-05-05 10:50:00 | 1392.92 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-05-05 10:40:00 | 1387.30 | 2026-05-05 15:20:00 | 1429.00 | TARGET_HIT | 0.50 | 3.01% |
| BUY | retest1 | 2026-05-08 10:00:00 | 1499.40 | 2026-05-08 10:05:00 | 1506.59 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-05-08 10:00:00 | 1499.40 | 2026-05-08 11:40:00 | 1499.40 | STOP_HIT | 0.50 | 0.00% |
