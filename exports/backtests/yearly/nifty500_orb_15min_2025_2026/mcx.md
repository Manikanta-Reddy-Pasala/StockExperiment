# Multi Commodity Exchange of India Ltd. (MCX)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2025-08-08 15:25:00 (4875 bars)
- **Last close:** 1536.90
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
| TARGET_HIT | 2 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 31 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 21
- **Target hits / Stop hits / Partials:** 2 / 21 / 8
- **Avg / median % per leg:** 0.09% / 0.00%
- **Sum % (uncompounded):** 2.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 4 | 33.3% | 1 | 8 | 3 | 0.24% | 2.9% |
| BUY @ 2nd Alert (retest1) | 12 | 4 | 33.3% | 1 | 8 | 3 | 0.24% | 2.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 19 | 6 | 31.6% | 1 | 13 | 5 | -0.00% | -0.0% |
| SELL @ 2nd Alert (retest1) | 19 | 6 | 31.6% | 1 | 13 | 5 | -0.00% | -0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 31 | 10 | 32.3% | 2 | 21 | 8 | 0.09% | 2.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 10:30:00 | 1204.20 | 1191.78 | 0.00 | ORB-long ORB[1186.30,1199.70] vol=2.3x ATR=4.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-14 10:35:00 | 1210.31 | 1196.51 | 0.00 | T1 1.5R @ 1210.31 |
| Target hit | 2025-05-14 15:20:00 | 1240.80 | 1218.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2025-05-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-19 09:35:00 | 1274.90 | 1288.86 | 0.00 | ORB-short ORB[1281.10,1298.80] vol=2.0x ATR=5.38 |
| Stop hit — per-position SL triggered | 2025-05-19 09:40:00 | 1280.28 | 1287.38 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-22 11:05:00 | 1256.00 | 1266.59 | 0.00 | ORB-short ORB[1260.20,1278.00] vol=2.0x ATR=4.78 |
| Stop hit — per-position SL triggered | 2025-05-22 11:25:00 | 1260.78 | 1265.11 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 10:00:00 | 1285.80 | 1273.97 | 0.00 | ORB-long ORB[1267.70,1279.90] vol=3.1x ATR=4.93 |
| Stop hit — per-position SL triggered | 2025-05-23 10:05:00 | 1280.87 | 1274.58 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-05-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-26 09:40:00 | 1299.50 | 1305.79 | 0.00 | ORB-short ORB[1302.00,1312.60] vol=1.7x ATR=4.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-26 10:35:00 | 1292.35 | 1301.88 | 0.00 | T1 1.5R @ 1292.35 |
| Stop hit — per-position SL triggered | 2025-05-26 10:50:00 | 1299.50 | 1301.24 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-05-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-29 09:30:00 | 1303.10 | 1301.38 | 0.00 | ORB-long ORB[1292.90,1303.00] vol=3.4x ATR=3.01 |
| Stop hit — per-position SL triggered | 2025-05-29 09:40:00 | 1300.09 | 1301.38 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-02 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-02 10:20:00 | 1314.90 | 1324.08 | 0.00 | ORB-short ORB[1316.90,1329.80] vol=1.9x ATR=3.78 |
| Stop hit — per-position SL triggered | 2025-06-02 10:25:00 | 1318.68 | 1323.76 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 10:00:00 | 1402.30 | 1389.01 | 0.00 | ORB-long ORB[1380.40,1391.00] vol=3.7x ATR=4.91 |
| Stop hit — per-position SL triggered | 2025-06-05 10:10:00 | 1397.39 | 1391.23 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-18 09:50:00 | 1592.00 | 1583.75 | 0.00 | ORB-long ORB[1576.20,1591.90] vol=1.8x ATR=6.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-18 10:05:00 | 1601.42 | 1587.85 | 0.00 | T1 1.5R @ 1601.42 |
| Stop hit — per-position SL triggered | 2025-06-18 10:35:00 | 1592.00 | 1589.68 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-07-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 09:30:00 | 1809.60 | 1801.47 | 0.00 | ORB-long ORB[1787.40,1808.30] vol=2.0x ATR=5.68 |
| Stop hit — per-position SL triggered | 2025-07-03 09:40:00 | 1803.92 | 1803.37 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-07-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-07 11:10:00 | 1761.20 | 1774.63 | 0.00 | ORB-short ORB[1764.70,1784.00] vol=2.4x ATR=4.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 11:45:00 | 1754.10 | 1769.87 | 0.00 | T1 1.5R @ 1754.10 |
| Target hit | 2025-07-07 15:20:00 | 1750.10 | 1759.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2025-07-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 09:30:00 | 1652.40 | 1656.86 | 0.00 | ORB-short ORB[1653.20,1664.00] vol=1.5x ATR=3.61 |
| Stop hit — per-position SL triggered | 2025-07-11 09:40:00 | 1656.01 | 1656.13 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-07-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 09:40:00 | 1642.60 | 1628.90 | 0.00 | ORB-long ORB[1613.60,1636.20] vol=2.7x ATR=7.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 09:45:00 | 1653.29 | 1633.33 | 0.00 | T1 1.5R @ 1653.29 |
| Stop hit — per-position SL triggered | 2025-07-14 11:10:00 | 1642.60 | 1643.72 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 09:30:00 | 1662.10 | 1657.91 | 0.00 | ORB-long ORB[1648.50,1661.70] vol=1.7x ATR=4.70 |
| Stop hit — per-position SL triggered | 2025-07-15 09:35:00 | 1657.40 | 1658.20 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 10:40:00 | 1662.90 | 1666.36 | 0.00 | ORB-short ORB[1663.40,1674.10] vol=1.8x ATR=4.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-16 11:45:00 | 1656.59 | 1665.22 | 0.00 | T1 1.5R @ 1656.59 |
| Stop hit — per-position SL triggered | 2025-07-16 12:20:00 | 1662.90 | 1664.69 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 09:35:00 | 1653.50 | 1645.24 | 0.00 | ORB-long ORB[1635.30,1648.90] vol=2.4x ATR=4.82 |
| Stop hit — per-position SL triggered | 2025-07-21 09:45:00 | 1648.68 | 1646.63 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 09:35:00 | 1646.00 | 1651.95 | 0.00 | ORB-short ORB[1651.80,1668.70] vol=6.0x ATR=4.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 09:40:00 | 1639.02 | 1651.50 | 0.00 | T1 1.5R @ 1639.02 |
| Stop hit — per-position SL triggered | 2025-07-22 09:45:00 | 1646.00 | 1651.37 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 09:30:00 | 1623.70 | 1632.02 | 0.00 | ORB-short ORB[1626.00,1642.00] vol=1.8x ATR=5.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-23 09:35:00 | 1616.18 | 1627.06 | 0.00 | T1 1.5R @ 1616.18 |
| Stop hit — per-position SL triggered | 2025-07-23 09:40:00 | 1623.70 | 1626.59 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 09:30:00 | 1637.30 | 1644.87 | 0.00 | ORB-short ORB[1642.60,1653.90] vol=1.7x ATR=3.99 |
| Stop hit — per-position SL triggered | 2025-07-24 09:40:00 | 1641.29 | 1643.88 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 09:45:00 | 1620.30 | 1631.95 | 0.00 | ORB-short ORB[1630.50,1646.40] vol=4.2x ATR=5.68 |
| Stop hit — per-position SL triggered | 2025-07-25 09:55:00 | 1625.98 | 1629.46 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-29 11:15:00 | 1530.30 | 1541.32 | 0.00 | ORB-short ORB[1544.80,1559.50] vol=2.8x ATR=5.15 |
| Stop hit — per-position SL triggered | 2025-07-29 11:20:00 | 1535.45 | 1540.99 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-07-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 09:45:00 | 1563.60 | 1574.55 | 0.00 | ORB-short ORB[1570.30,1581.70] vol=1.7x ATR=5.01 |
| Stop hit — per-position SL triggered | 2025-07-30 09:55:00 | 1568.61 | 1573.35 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-08-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-01 09:35:00 | 1522.40 | 1530.05 | 0.00 | ORB-short ORB[1523.10,1543.80] vol=1.9x ATR=5.11 |
| Stop hit — per-position SL triggered | 2025-08-01 09:40:00 | 1527.51 | 1529.54 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 10:30:00 | 1204.20 | 2025-05-14 10:35:00 | 1210.31 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-05-14 10:30:00 | 1204.20 | 2025-05-14 15:20:00 | 1240.80 | TARGET_HIT | 0.50 | 3.04% |
| SELL | retest1 | 2025-05-19 09:35:00 | 1274.90 | 2025-05-19 09:40:00 | 1280.28 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-05-22 11:05:00 | 1256.00 | 2025-05-22 11:25:00 | 1260.78 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-05-23 10:00:00 | 1285.80 | 2025-05-23 10:05:00 | 1280.87 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-05-26 09:40:00 | 1299.50 | 2025-05-26 10:35:00 | 1292.35 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-05-26 09:40:00 | 1299.50 | 2025-05-26 10:50:00 | 1299.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-29 09:30:00 | 1303.10 | 2025-05-29 09:40:00 | 1300.09 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-06-02 10:20:00 | 1314.90 | 2025-06-02 10:25:00 | 1318.68 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-06-05 10:00:00 | 1402.30 | 2025-06-05 10:10:00 | 1397.39 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-06-18 09:50:00 | 1592.00 | 2025-06-18 10:05:00 | 1601.42 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-06-18 09:50:00 | 1592.00 | 2025-06-18 10:35:00 | 1592.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-03 09:30:00 | 1809.60 | 2025-07-03 09:40:00 | 1803.92 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-07-07 11:10:00 | 1761.20 | 2025-07-07 11:45:00 | 1754.10 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-07-07 11:10:00 | 1761.20 | 2025-07-07 15:20:00 | 1750.10 | TARGET_HIT | 0.50 | 0.63% |
| SELL | retest1 | 2025-07-11 09:30:00 | 1652.40 | 2025-07-11 09:40:00 | 1656.01 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-07-14 09:40:00 | 1642.60 | 2025-07-14 09:45:00 | 1653.29 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2025-07-14 09:40:00 | 1642.60 | 2025-07-14 11:10:00 | 1642.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-15 09:30:00 | 1662.10 | 2025-07-15 09:35:00 | 1657.40 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-07-16 10:40:00 | 1662.90 | 2025-07-16 11:45:00 | 1656.59 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-07-16 10:40:00 | 1662.90 | 2025-07-16 12:20:00 | 1662.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-21 09:35:00 | 1653.50 | 2025-07-21 09:45:00 | 1648.68 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-07-22 09:35:00 | 1646.00 | 2025-07-22 09:40:00 | 1639.02 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-07-22 09:35:00 | 1646.00 | 2025-07-22 09:45:00 | 1646.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-23 09:30:00 | 1623.70 | 2025-07-23 09:35:00 | 1616.18 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-07-23 09:30:00 | 1623.70 | 2025-07-23 09:40:00 | 1623.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-24 09:30:00 | 1637.30 | 2025-07-24 09:40:00 | 1641.29 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-07-25 09:45:00 | 1620.30 | 2025-07-25 09:55:00 | 1625.98 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-07-29 11:15:00 | 1530.30 | 2025-07-29 11:20:00 | 1535.45 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-07-30 09:45:00 | 1563.60 | 2025-07-30 09:55:00 | 1568.61 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-08-01 09:35:00 | 1522.40 | 2025-08-01 09:40:00 | 1527.51 | STOP_HIT | 1.00 | -0.34% |
