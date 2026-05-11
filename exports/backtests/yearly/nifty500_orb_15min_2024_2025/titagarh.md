# Titagarh Rail Systems Ltd. (TITAGARH)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 840.00
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
| ENTRY1 | 23 |
| ENTRY2 | 0 |
| PARTIAL | 8 |
| TARGET_HIT | 3 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 31 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 20
- **Target hits / Stop hits / Partials:** 3 / 20 / 8
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 5.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 5 | 41.7% | 1 | 7 | 4 | 0.42% | 5.0% |
| BUY @ 2nd Alert (retest1) | 12 | 5 | 41.7% | 1 | 7 | 4 | 0.42% | 5.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 19 | 6 | 31.6% | 2 | 13 | 4 | 0.01% | 0.1% |
| SELL @ 2nd Alert (retest1) | 19 | 6 | 31.6% | 2 | 13 | 4 | 0.01% | 0.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 31 | 11 | 35.5% | 3 | 20 | 8 | 0.17% | 5.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:30:00 | 1527.90 | 1513.26 | 0.00 | ORB-long ORB[1495.45,1515.00] vol=4.4x ATR=7.96 |
| Stop hit — per-position SL triggered | 2024-06-21 09:35:00 | 1519.94 | 1518.65 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-07-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 09:30:00 | 1854.90 | 1843.02 | 0.00 | ORB-long ORB[1829.60,1852.00] vol=2.5x ATR=7.64 |
| Stop hit — per-position SL triggered | 2024-07-02 09:40:00 | 1847.26 | 1847.71 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-07-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 10:35:00 | 1825.00 | 1791.91 | 0.00 | ORB-long ORB[1780.00,1800.00] vol=5.2x ATR=8.25 |
| Stop hit — per-position SL triggered | 2024-07-05 10:40:00 | 1816.75 | 1800.78 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 11:15:00 | 1621.35 | 1630.20 | 0.00 | ORB-short ORB[1623.10,1645.60] vol=2.2x ATR=5.94 |
| Stop hit — per-position SL triggered | 2024-07-23 11:25:00 | 1627.29 | 1629.98 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-07-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-26 10:20:00 | 1577.10 | 1589.08 | 0.00 | ORB-short ORB[1586.20,1608.70] vol=2.3x ATR=8.10 |
| Stop hit — per-position SL triggered | 2024-07-26 10:40:00 | 1585.20 | 1588.09 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-08-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-14 09:45:00 | 1422.70 | 1414.65 | 0.00 | ORB-long ORB[1387.45,1408.70] vol=5.2x ATR=11.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 09:55:00 | 1439.45 | 1419.20 | 0.00 | T1 1.5R @ 1439.45 |
| Stop hit — per-position SL triggered | 2024-08-14 10:05:00 | 1422.70 | 1420.42 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-08-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:30:00 | 1458.00 | 1445.78 | 0.00 | ORB-long ORB[1428.95,1449.00] vol=2.4x ATR=5.71 |
| Stop hit — per-position SL triggered | 2024-08-21 09:35:00 | 1452.29 | 1447.62 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-08-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-22 09:35:00 | 1432.00 | 1438.63 | 0.00 | ORB-short ORB[1436.00,1446.95] vol=1.9x ATR=4.15 |
| Stop hit — per-position SL triggered | 2024-08-22 09:40:00 | 1436.15 | 1437.91 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-08-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-26 10:00:00 | 1401.80 | 1407.05 | 0.00 | ORB-short ORB[1406.05,1421.95] vol=2.1x ATR=4.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 11:05:00 | 1394.62 | 1403.68 | 0.00 | T1 1.5R @ 1394.62 |
| Stop hit — per-position SL triggered | 2024-08-26 11:20:00 | 1401.80 | 1402.70 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-09-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-11 09:30:00 | 1377.75 | 1383.48 | 0.00 | ORB-short ORB[1378.00,1397.25] vol=1.6x ATR=4.50 |
| Stop hit — per-position SL triggered | 2024-09-11 09:40:00 | 1382.25 | 1382.63 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-09-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-12 10:55:00 | 1347.75 | 1356.75 | 0.00 | ORB-short ORB[1357.15,1375.00] vol=1.5x ATR=4.30 |
| Stop hit — per-position SL triggered | 2024-09-12 11:05:00 | 1352.05 | 1356.54 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-09-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 09:35:00 | 1321.35 | 1329.87 | 0.00 | ORB-short ORB[1325.15,1345.00] vol=2.1x ATR=4.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 09:55:00 | 1313.86 | 1325.84 | 0.00 | T1 1.5R @ 1313.86 |
| Target hit | 2024-09-17 15:20:00 | 1302.90 | 1313.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2024-09-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:50:00 | 1257.65 | 1275.05 | 0.00 | ORB-short ORB[1278.00,1295.85] vol=1.5x ATR=6.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:10:00 | 1247.18 | 1268.17 | 0.00 | T1 1.5R @ 1247.18 |
| Stop hit — per-position SL triggered | 2024-09-19 10:30:00 | 1257.65 | 1265.08 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-10-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:55:00 | 1119.30 | 1101.02 | 0.00 | ORB-long ORB[1076.00,1091.95] vol=2.9x ATR=7.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 10:00:00 | 1129.95 | 1109.39 | 0.00 | T1 1.5R @ 1129.95 |
| Stop hit — per-position SL triggered | 2024-10-11 10:15:00 | 1119.30 | 1120.98 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-11-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 09:40:00 | 1165.80 | 1152.79 | 0.00 | ORB-long ORB[1140.55,1156.65] vol=2.4x ATR=5.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-27 09:45:00 | 1173.62 | 1160.73 | 0.00 | T1 1.5R @ 1173.62 |
| Target hit | 2024-11-27 15:20:00 | 1206.05 | 1190.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2024-12-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 09:45:00 | 1182.05 | 1185.47 | 0.00 | ORB-short ORB[1183.00,1195.45] vol=1.7x ATR=4.83 |
| Stop hit — per-position SL triggered | 2024-12-06 10:50:00 | 1186.88 | 1183.76 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-01-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 11:00:00 | 1099.05 | 1106.38 | 0.00 | ORB-short ORB[1100.10,1110.00] vol=2.3x ATR=3.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 13:35:00 | 1093.28 | 1102.79 | 0.00 | T1 1.5R @ 1093.28 |
| Target hit | 2025-01-09 15:20:00 | 1088.25 | 1099.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — SELL (started 2025-03-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-06 09:30:00 | 719.25 | 725.35 | 0.00 | ORB-short ORB[722.35,732.50] vol=1.9x ATR=4.24 |
| Stop hit — per-position SL triggered | 2025-03-06 09:35:00 | 723.49 | 724.87 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-03-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 10:00:00 | 695.00 | 703.76 | 0.00 | ORB-short ORB[699.60,708.90] vol=2.1x ATR=3.37 |
| Stop hit — per-position SL triggered | 2025-03-12 10:35:00 | 698.37 | 701.41 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-03-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-28 11:10:00 | 812.60 | 823.68 | 0.00 | ORB-short ORB[819.55,831.00] vol=1.9x ATR=3.67 |
| Stop hit — per-position SL triggered | 2025-03-28 11:15:00 | 816.27 | 823.42 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-04-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:45:00 | 798.55 | 792.06 | 0.00 | ORB-long ORB[787.25,796.80] vol=1.8x ATR=2.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 09:50:00 | 802.89 | 794.02 | 0.00 | T1 1.5R @ 802.89 |
| Stop hit — per-position SL triggered | 2025-04-21 10:10:00 | 798.55 | 796.62 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 09:35:00 | 798.85 | 804.66 | 0.00 | ORB-short ORB[799.05,810.50] vol=2.1x ATR=3.34 |
| Stop hit — per-position SL triggered | 2025-04-23 09:45:00 | 802.19 | 804.32 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-04-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 09:40:00 | 779.10 | 787.77 | 0.00 | ORB-short ORB[783.10,794.40] vol=2.1x ATR=4.05 |
| Stop hit — per-position SL triggered | 2025-04-29 09:45:00 | 783.15 | 787.09 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-06-21 09:30:00 | 1527.90 | 2024-06-21 09:35:00 | 1519.94 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2024-07-02 09:30:00 | 1854.90 | 2024-07-02 09:40:00 | 1847.26 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-07-05 10:35:00 | 1825.00 | 2024-07-05 10:40:00 | 1816.75 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-07-23 11:15:00 | 1621.35 | 2024-07-23 11:25:00 | 1627.29 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-07-26 10:20:00 | 1577.10 | 2024-07-26 10:40:00 | 1585.20 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2024-08-14 09:45:00 | 1422.70 | 2024-08-14 09:55:00 | 1439.45 | PARTIAL | 0.50 | 1.18% |
| BUY | retest1 | 2024-08-14 09:45:00 | 1422.70 | 2024-08-14 10:05:00 | 1422.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-21 09:30:00 | 1458.00 | 2024-08-21 09:35:00 | 1452.29 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-08-22 09:35:00 | 1432.00 | 2024-08-22 09:40:00 | 1436.15 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-08-26 10:00:00 | 1401.80 | 2024-08-26 11:05:00 | 1394.62 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-08-26 10:00:00 | 1401.80 | 2024-08-26 11:20:00 | 1401.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-11 09:30:00 | 1377.75 | 2024-09-11 09:40:00 | 1382.25 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-09-12 10:55:00 | 1347.75 | 2024-09-12 11:05:00 | 1352.05 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-09-17 09:35:00 | 1321.35 | 2024-09-17 09:55:00 | 1313.86 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-09-17 09:35:00 | 1321.35 | 2024-09-17 15:20:00 | 1302.90 | TARGET_HIT | 0.50 | 1.40% |
| SELL | retest1 | 2024-09-19 09:50:00 | 1257.65 | 2024-09-19 10:10:00 | 1247.18 | PARTIAL | 0.50 | 0.83% |
| SELL | retest1 | 2024-09-19 09:50:00 | 1257.65 | 2024-09-19 10:30:00 | 1257.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-11 09:55:00 | 1119.30 | 2024-10-11 10:00:00 | 1129.95 | PARTIAL | 0.50 | 0.95% |
| BUY | retest1 | 2024-10-11 09:55:00 | 1119.30 | 2024-10-11 10:15:00 | 1119.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-27 09:40:00 | 1165.80 | 2024-11-27 09:45:00 | 1173.62 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2024-11-27 09:40:00 | 1165.80 | 2024-11-27 15:20:00 | 1206.05 | TARGET_HIT | 0.50 | 3.45% |
| SELL | retest1 | 2024-12-06 09:45:00 | 1182.05 | 2024-12-06 10:50:00 | 1186.88 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-01-09 11:00:00 | 1099.05 | 2025-01-09 13:35:00 | 1093.28 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-01-09 11:00:00 | 1099.05 | 2025-01-09 15:20:00 | 1088.25 | TARGET_HIT | 0.50 | 0.98% |
| SELL | retest1 | 2025-03-06 09:30:00 | 719.25 | 2025-03-06 09:35:00 | 723.49 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest1 | 2025-03-12 10:00:00 | 695.00 | 2025-03-12 10:35:00 | 698.37 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2025-03-28 11:10:00 | 812.60 | 2025-03-28 11:15:00 | 816.27 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-04-21 09:45:00 | 798.55 | 2025-04-21 09:50:00 | 802.89 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-04-21 09:45:00 | 798.55 | 2025-04-21 10:10:00 | 798.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-23 09:35:00 | 798.85 | 2025-04-23 09:45:00 | 802.19 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-04-29 09:40:00 | 779.10 | 2025-04-29 09:45:00 | 783.15 | STOP_HIT | 1.00 | -0.52% |
