# DOMS Industries Ltd. (DOMS)

## Backtest Summary

- **Window:** 2023-12-20 09:40:00 → 2026-05-08 15:25:00 (42662 bars)
- **Last close:** 2340.00
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
| TARGET_HIT | 1 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 16
- **Target hits / Stop hits / Partials:** 1 / 16 / 5
- **Avg / median % per leg:** 0.01% / -0.21%
- **Sum % (uncompounded):** 0.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 2 | 25.0% | 1 | 6 | 1 | 0.01% | 0.0% |
| BUY @ 2nd Alert (retest1) | 8 | 2 | 25.0% | 1 | 6 | 1 | 0.01% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 4 | 28.6% | 0 | 10 | 4 | 0.01% | 0.1% |
| SELL @ 2nd Alert (retest1) | 14 | 4 | 28.6% | 0 | 10 | 4 | 0.01% | 0.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 6 | 27.3% | 1 | 16 | 5 | 0.01% | 0.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-12-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-28 09:30:00 | 1283.15 | 1288.23 | 0.00 | ORB-short ORB[1285.00,1295.00] vol=1.8x ATR=4.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-28 10:40:00 | 1276.89 | 1283.52 | 0.00 | T1 1.5R @ 1276.89 |
| Stop hit — per-position SL triggered | 2023-12-28 15:10:00 | 1283.15 | 1272.60 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-12-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-29 11:05:00 | 1255.90 | 1262.65 | 0.00 | ORB-short ORB[1261.80,1273.00] vol=2.0x ATR=3.80 |
| Stop hit — per-position SL triggered | 2023-12-29 11:20:00 | 1259.70 | 1262.40 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-01-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 10:00:00 | 1263.30 | 1271.90 | 0.00 | ORB-short ORB[1270.80,1283.85] vol=2.5x ATR=3.65 |
| Stop hit — per-position SL triggered | 2024-01-02 10:10:00 | 1266.95 | 1271.96 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-01-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-03 11:05:00 | 1287.55 | 1279.96 | 0.00 | ORB-long ORB[1271.00,1285.00] vol=1.6x ATR=3.27 |
| Stop hit — per-position SL triggered | 2024-01-03 11:20:00 | 1284.28 | 1281.25 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-01-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 10:00:00 | 1320.55 | 1317.55 | 0.00 | ORB-long ORB[1313.50,1320.00] vol=2.0x ATR=3.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-09 11:00:00 | 1325.21 | 1321.46 | 0.00 | T1 1.5R @ 1325.21 |
| Target hit | 2024-01-09 15:20:00 | 1352.00 | 1339.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2024-01-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-19 09:45:00 | 1445.00 | 1430.99 | 0.00 | ORB-long ORB[1420.80,1439.00] vol=2.5x ATR=7.71 |
| Stop hit — per-position SL triggered | 2024-01-19 10:00:00 | 1437.29 | 1435.81 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-01-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-25 09:50:00 | 1422.60 | 1428.97 | 0.00 | ORB-short ORB[1424.60,1442.00] vol=1.7x ATR=5.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-25 10:15:00 | 1414.43 | 1427.49 | 0.00 | T1 1.5R @ 1414.43 |
| Stop hit — per-position SL triggered | 2024-01-25 11:30:00 | 1422.60 | 1421.84 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-30 10:15:00 | 1439.60 | 1446.79 | 0.00 | ORB-short ORB[1443.00,1461.00] vol=2.5x ATR=7.75 |
| Stop hit — per-position SL triggered | 2024-01-30 10:30:00 | 1447.35 | 1443.22 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-02-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-07 09:40:00 | 1438.75 | 1444.95 | 0.00 | ORB-short ORB[1440.00,1455.95] vol=2.0x ATR=4.10 |
| Stop hit — per-position SL triggered | 2024-02-07 09:45:00 | 1442.85 | 1444.86 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-02-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-16 10:40:00 | 1550.50 | 1570.28 | 0.00 | ORB-short ORB[1571.00,1590.00] vol=2.1x ATR=6.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-16 10:50:00 | 1540.73 | 1569.01 | 0.00 | T1 1.5R @ 1540.73 |
| Stop hit — per-position SL triggered | 2024-02-16 11:15:00 | 1550.50 | 1567.93 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-19 09:30:00 | 1578.90 | 1561.74 | 0.00 | ORB-long ORB[1545.15,1568.00] vol=2.6x ATR=7.87 |
| Stop hit — per-position SL triggered | 2024-02-19 09:40:00 | 1571.03 | 1562.16 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-03-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-13 09:30:00 | 1396.80 | 1407.50 | 0.00 | ORB-short ORB[1399.30,1419.65] vol=1.5x ATR=6.32 |
| Stop hit — per-position SL triggered | 2024-03-13 09:35:00 | 1403.12 | 1405.54 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-03-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 09:55:00 | 1503.05 | 1496.52 | 0.00 | ORB-long ORB[1475.00,1497.00] vol=5.6x ATR=10.73 |
| Stop hit — per-position SL triggered | 2024-03-22 13:50:00 | 1492.32 | 1497.64 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-04-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-09 09:40:00 | 1702.00 | 1698.39 | 0.00 | ORB-long ORB[1684.50,1701.90] vol=1.7x ATR=6.17 |
| Stop hit — per-position SL triggered | 2024-04-09 13:35:00 | 1695.83 | 1700.67 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-02 11:15:00 | 1824.75 | 1836.15 | 0.00 | ORB-short ORB[1825.00,1844.40] vol=1.6x ATR=3.83 |
| Stop hit — per-position SL triggered | 2024-05-02 11:20:00 | 1828.58 | 1836.04 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-05-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-03 09:55:00 | 1856.35 | 1847.96 | 0.00 | ORB-long ORB[1836.00,1849.85] vol=2.8x ATR=6.10 |
| Stop hit — per-position SL triggered | 2024-05-03 10:05:00 | 1850.25 | 1850.97 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-05-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-07 10:35:00 | 1777.15 | 1789.43 | 0.00 | ORB-short ORB[1793.20,1812.90] vol=2.4x ATR=5.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 10:45:00 | 1768.38 | 1785.93 | 0.00 | T1 1.5R @ 1768.38 |
| Stop hit — per-position SL triggered | 2024-05-07 12:30:00 | 1777.15 | 1779.97 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-12-28 09:30:00 | 1283.15 | 2023-12-28 10:40:00 | 1276.89 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2023-12-28 09:30:00 | 1283.15 | 2023-12-28 15:10:00 | 1283.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-29 11:05:00 | 1255.90 | 2023-12-29 11:20:00 | 1259.70 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-01-02 10:00:00 | 1263.30 | 2024-01-02 10:10:00 | 1266.95 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-01-03 11:05:00 | 1287.55 | 2024-01-03 11:20:00 | 1284.28 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-01-09 10:00:00 | 1320.55 | 2024-01-09 11:00:00 | 1325.21 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-01-09 10:00:00 | 1320.55 | 2024-01-09 15:20:00 | 1352.00 | TARGET_HIT | 0.50 | 2.38% |
| BUY | retest1 | 2024-01-19 09:45:00 | 1445.00 | 2024-01-19 10:00:00 | 1437.29 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2024-01-25 09:50:00 | 1422.60 | 2024-01-25 10:15:00 | 1414.43 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-01-25 09:50:00 | 1422.60 | 2024-01-25 11:30:00 | 1422.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-30 10:15:00 | 1439.60 | 2024-01-30 10:30:00 | 1447.35 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2024-02-07 09:40:00 | 1438.75 | 2024-02-07 09:45:00 | 1442.85 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-02-16 10:40:00 | 1550.50 | 2024-02-16 10:50:00 | 1540.73 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-02-16 10:40:00 | 1550.50 | 2024-02-16 11:15:00 | 1550.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-19 09:30:00 | 1578.90 | 2024-02-19 09:40:00 | 1571.03 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-03-13 09:30:00 | 1396.80 | 2024-03-13 09:35:00 | 1403.12 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-03-22 09:55:00 | 1503.05 | 2024-03-22 13:50:00 | 1492.32 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest1 | 2024-04-09 09:40:00 | 1702.00 | 2024-04-09 13:35:00 | 1695.83 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-05-02 11:15:00 | 1824.75 | 2024-05-02 11:20:00 | 1828.58 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-05-03 09:55:00 | 1856.35 | 2024-05-03 10:05:00 | 1850.25 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-05-07 10:35:00 | 1777.15 | 2024-05-07 10:45:00 | 1768.38 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-05-07 10:35:00 | 1777.15 | 2024-05-07 12:30:00 | 1777.15 | STOP_HIT | 0.50 | 0.00% |
