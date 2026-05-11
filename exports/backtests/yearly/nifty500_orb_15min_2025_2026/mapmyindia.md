# C.E. Info Systems Ltd. (MAPMYINDIA)

## Backtest Summary

- **Window:** 2025-11-07 09:15:00 → 2026-05-05 15:25:00 (9000 bars)
- **Last close:** 942.20
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
| ENTRY1 | 35 |
| ENTRY2 | 0 |
| PARTIAL | 16 |
| TARGET_HIT | 10 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 51 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 25
- **Target hits / Stop hits / Partials:** 10 / 25 / 16
- **Avg / median % per leg:** 0.24% / 0.17%
- **Sum % (uncompounded):** 12.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 11 | 50.0% | 4 | 11 | 7 | 0.18% | 3.9% |
| BUY @ 2nd Alert (retest1) | 22 | 11 | 50.0% | 4 | 11 | 7 | 0.18% | 3.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 29 | 15 | 51.7% | 6 | 14 | 9 | 0.29% | 8.4% |
| SELL @ 2nd Alert (retest1) | 29 | 15 | 51.7% | 6 | 14 | 9 | 0.29% | 8.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 51 | 26 | 51.0% | 10 | 25 | 16 | 0.24% | 12.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-11-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 11:05:00 | 1744.30 | 1730.56 | 0.00 | ORB-long ORB[1724.80,1739.40] vol=4.5x ATR=5.34 |
| Stop hit — per-position SL triggered | 2025-11-12 12:55:00 | 1738.96 | 1735.55 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-11-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 10:55:00 | 1701.00 | 1691.84 | 0.00 | ORB-long ORB[1687.40,1697.90] vol=1.7x ATR=3.75 |
| Stop hit — per-position SL triggered | 2025-11-14 11:20:00 | 1697.25 | 1694.51 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-11-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 10:05:00 | 1687.80 | 1692.19 | 0.00 | ORB-short ORB[1693.70,1704.90] vol=3.7x ATR=4.46 |
| Stop hit — per-position SL triggered | 2025-11-18 10:10:00 | 1692.26 | 1692.13 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-11-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-25 10:55:00 | 1663.00 | 1667.46 | 0.00 | ORB-short ORB[1667.60,1686.60] vol=3.0x ATR=4.19 |
| Stop hit — per-position SL triggered | 2025-11-25 11:15:00 | 1667.19 | 1667.17 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-11-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 09:30:00 | 1677.00 | 1668.75 | 0.00 | ORB-long ORB[1657.20,1668.90] vol=5.5x ATR=5.46 |
| Stop hit — per-position SL triggered | 2025-11-26 09:45:00 | 1671.54 | 1671.89 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-11-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 10:20:00 | 1665.90 | 1673.65 | 0.00 | ORB-short ORB[1675.20,1684.90] vol=1.6x ATR=4.46 |
| Stop hit — per-position SL triggered | 2025-11-27 11:10:00 | 1670.36 | 1671.65 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-11-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-28 09:35:00 | 1648.00 | 1652.96 | 0.00 | ORB-short ORB[1651.00,1667.90] vol=1.7x ATR=4.67 |
| Stop hit — per-position SL triggered | 2025-11-28 09:45:00 | 1652.67 | 1655.32 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-12-05 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 10:30:00 | 1660.80 | 1666.56 | 0.00 | ORB-short ORB[1667.10,1687.60] vol=2.0x ATR=4.36 |
| Stop hit — per-position SL triggered | 2025-12-05 11:15:00 | 1665.16 | 1666.37 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-12-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:50:00 | 1656.00 | 1660.82 | 0.00 | ORB-short ORB[1660.00,1672.40] vol=2.0x ATR=3.58 |
| Stop hit — per-position SL triggered | 2025-12-08 10:55:00 | 1659.58 | 1660.66 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-12-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 10:55:00 | 1623.60 | 1638.92 | 0.00 | ORB-short ORB[1636.70,1650.00] vol=1.9x ATR=5.61 |
| Stop hit — per-position SL triggered | 2025-12-10 11:05:00 | 1629.21 | 1637.32 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-12-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 09:50:00 | 1664.20 | 1650.25 | 0.00 | ORB-long ORB[1637.10,1650.20] vol=2.1x ATR=5.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 09:55:00 | 1672.79 | 1657.09 | 0.00 | T1 1.5R @ 1672.79 |
| Target hit | 2025-12-15 12:20:00 | 1667.10 | 1673.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — BUY (started 2025-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 11:15:00 | 1637.00 | 1626.83 | 0.00 | ORB-long ORB[1617.20,1630.80] vol=4.5x ATR=4.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 11:30:00 | 1643.21 | 1630.34 | 0.00 | T1 1.5R @ 1643.21 |
| Target hit | 2025-12-18 15:20:00 | 1655.90 | 1653.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2025-12-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 10:40:00 | 1695.60 | 1678.91 | 0.00 | ORB-long ORB[1652.10,1676.00] vol=2.0x ATR=6.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-22 10:55:00 | 1705.50 | 1683.55 | 0.00 | T1 1.5R @ 1705.50 |
| Stop hit — per-position SL triggered | 2025-12-22 11:20:00 | 1695.60 | 1687.53 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-12-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 09:55:00 | 1654.90 | 1664.52 | 0.00 | ORB-short ORB[1664.50,1684.40] vol=1.6x ATR=5.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 10:45:00 | 1646.48 | 1658.33 | 0.00 | T1 1.5R @ 1646.48 |
| Stop hit — per-position SL triggered | 2025-12-29 13:50:00 | 1654.90 | 1655.21 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-01-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-01 11:05:00 | 1710.90 | 1711.77 | 0.00 | ORB-short ORB[1711.70,1720.90] vol=4.4x ATR=3.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 11:20:00 | 1705.56 | 1711.60 | 0.00 | T1 1.5R @ 1705.56 |
| Stop hit — per-position SL triggered | 2026-01-01 12:30:00 | 1710.90 | 1710.69 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-01-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-02 11:05:00 | 1711.00 | 1716.22 | 0.00 | ORB-short ORB[1713.30,1721.30] vol=4.1x ATR=3.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 11:25:00 | 1705.03 | 1712.82 | 0.00 | T1 1.5R @ 1705.03 |
| Target hit | 2026-01-02 15:10:00 | 1705.70 | 1701.32 | 0.00 | Trail-exit close>VWAP |

### Cycle 17 — SELL (started 2026-01-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:45:00 | 1669.30 | 1677.82 | 0.00 | ORB-short ORB[1682.00,1695.00] vol=3.9x ATR=3.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:15:00 | 1663.39 | 1675.39 | 0.00 | T1 1.5R @ 1663.39 |
| Stop hit — per-position SL triggered | 2026-01-08 11:25:00 | 1669.30 | 1675.13 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-01-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-09 10:50:00 | 1622.20 | 1626.95 | 0.00 | ORB-short ORB[1624.20,1635.90] vol=1.7x ATR=4.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 11:25:00 | 1616.18 | 1625.79 | 0.00 | T1 1.5R @ 1616.18 |
| Target hit | 2026-01-09 14:55:00 | 1604.00 | 1603.62 | 0.00 | Trail-exit close>VWAP |

### Cycle 19 — SELL (started 2026-01-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 10:00:00 | 1548.10 | 1557.10 | 0.00 | ORB-short ORB[1553.30,1565.20] vol=2.5x ATR=4.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 11:25:00 | 1540.68 | 1547.91 | 0.00 | T1 1.5R @ 1540.68 |
| Target hit | 2026-01-14 15:20:00 | 1525.00 | 1536.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2026-01-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-16 10:35:00 | 1522.90 | 1525.78 | 0.00 | ORB-short ORB[1524.90,1534.80] vol=2.8x ATR=3.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 10:40:00 | 1517.88 | 1524.45 | 0.00 | T1 1.5R @ 1517.88 |
| Target hit | 2026-01-16 12:05:00 | 1514.80 | 1511.78 | 0.00 | Trail-exit close>VWAP |

### Cycle 21 — SELL (started 2026-01-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-23 09:55:00 | 1335.50 | 1341.76 | 0.00 | ORB-short ORB[1339.50,1358.90] vol=1.9x ATR=4.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 11:35:00 | 1328.18 | 1336.40 | 0.00 | T1 1.5R @ 1328.18 |
| Target hit | 2026-01-23 15:20:00 | 1303.30 | 1318.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — SELL (started 2026-01-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 11:05:00 | 1285.50 | 1293.08 | 0.00 | ORB-short ORB[1296.90,1313.00] vol=13.1x ATR=4.44 |
| Stop hit — per-position SL triggered | 2026-01-28 11:20:00 | 1289.94 | 1292.68 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2026-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 09:30:00 | 1294.90 | 1287.94 | 0.00 | ORB-long ORB[1280.10,1294.60] vol=3.4x ATR=6.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 09:45:00 | 1304.57 | 1290.71 | 0.00 | T1 1.5R @ 1304.57 |
| Target hit | 2026-01-30 13:40:00 | 1306.60 | 1309.83 | 0.00 | Trail-exit close<VWAP |

### Cycle 24 — BUY (started 2026-02-01 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 10:25:00 | 1340.30 | 1329.70 | 0.00 | ORB-long ORB[1312.20,1324.00] vol=1.8x ATR=4.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 10:45:00 | 1347.27 | 1333.53 | 0.00 | T1 1.5R @ 1347.27 |
| Stop hit — per-position SL triggered | 2026-02-01 11:30:00 | 1340.30 | 1338.92 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2026-02-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:05:00 | 1295.40 | 1289.70 | 0.00 | ORB-long ORB[1278.30,1295.00] vol=2.7x ATR=4.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:10:00 | 1302.75 | 1294.38 | 0.00 | T1 1.5R @ 1302.75 |
| Target hit | 2026-02-10 11:30:00 | 1308.90 | 1311.98 | 0.00 | Trail-exit close<VWAP |

### Cycle 26 — SELL (started 2026-02-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:55:00 | 1103.30 | 1109.16 | 0.00 | ORB-short ORB[1108.00,1119.30] vol=2.3x ATR=3.57 |
| Stop hit — per-position SL triggered | 2026-02-23 11:00:00 | 1106.87 | 1109.02 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2026-02-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:45:00 | 1075.80 | 1062.32 | 0.00 | ORB-long ORB[1049.00,1061.90] vol=2.7x ATR=5.36 |
| Stop hit — per-position SL triggered | 2026-02-26 09:55:00 | 1070.44 | 1063.09 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2026-03-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 10:45:00 | 1009.90 | 1003.77 | 0.00 | ORB-long ORB[991.20,1003.00] vol=3.7x ATR=4.47 |
| Stop hit — per-position SL triggered | 2026-03-04 11:30:00 | 1005.43 | 1008.43 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2026-03-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:55:00 | 998.80 | 1003.39 | 0.00 | ORB-short ORB[1000.00,1009.00] vol=5.0x ATR=2.42 |
| Stop hit — per-position SL triggered | 2026-03-06 13:55:00 | 1001.22 | 1002.03 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2026-03-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:20:00 | 951.70 | 957.47 | 0.00 | ORB-short ORB[953.00,965.30] vol=2.1x ATR=4.53 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 956.23 | 956.90 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 11:15:00 | 885.30 | 876.21 | 0.00 | ORB-long ORB[868.90,878.70] vol=3.5x ATR=3.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 11:45:00 | 890.42 | 877.90 | 0.00 | T1 1.5R @ 890.42 |
| Stop hit — per-position SL triggered | 2026-03-20 11:50:00 | 885.30 | 878.77 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2026-03-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:00:00 | 893.80 | 887.93 | 0.00 | ORB-long ORB[879.70,890.50] vol=1.6x ATR=5.21 |
| Stop hit — per-position SL triggered | 2026-03-25 10:20:00 | 888.59 | 888.52 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2026-04-15 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:40:00 | 956.55 | 945.80 | 0.00 | ORB-long ORB[935.00,948.65] vol=2.9x ATR=4.65 |
| Stop hit — per-position SL triggered | 2026-04-15 10:50:00 | 951.90 | 946.48 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2026-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:30:00 | 939.60 | 946.48 | 0.00 | ORB-short ORB[943.40,955.90] vol=2.4x ATR=4.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 09:40:00 | 933.00 | 941.61 | 0.00 | T1 1.5R @ 933.00 |
| Target hit | 2026-04-16 14:45:00 | 922.55 | 922.03 | 0.00 | Trail-exit close>VWAP |

### Cycle 35 — BUY (started 2026-05-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:00:00 | 940.10 | 932.48 | 0.00 | ORB-long ORB[922.80,934.65] vol=1.9x ATR=4.24 |
| Stop hit — per-position SL triggered | 2026-05-04 10:10:00 | 935.86 | 933.07 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-11-12 11:05:00 | 1744.30 | 2025-11-12 12:55:00 | 1738.96 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-11-14 10:55:00 | 1701.00 | 2025-11-14 11:20:00 | 1697.25 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-11-18 10:05:00 | 1687.80 | 2025-11-18 10:10:00 | 1692.26 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-11-25 10:55:00 | 1663.00 | 2025-11-25 11:15:00 | 1667.19 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-11-26 09:30:00 | 1677.00 | 2025-11-26 09:45:00 | 1671.54 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-11-27 10:20:00 | 1665.90 | 2025-11-27 11:10:00 | 1670.36 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-11-28 09:35:00 | 1648.00 | 2025-11-28 09:45:00 | 1652.67 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-12-05 10:30:00 | 1660.80 | 2025-12-05 11:15:00 | 1665.16 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-12-08 10:50:00 | 1656.00 | 2025-12-08 10:55:00 | 1659.58 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-12-10 10:55:00 | 1623.60 | 2025-12-10 11:05:00 | 1629.21 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-12-15 09:50:00 | 1664.20 | 2025-12-15 09:55:00 | 1672.79 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-12-15 09:50:00 | 1664.20 | 2025-12-15 12:20:00 | 1667.10 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2025-12-18 11:15:00 | 1637.00 | 2025-12-18 11:30:00 | 1643.21 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-12-18 11:15:00 | 1637.00 | 2025-12-18 15:20:00 | 1655.90 | TARGET_HIT | 0.50 | 1.15% |
| BUY | retest1 | 2025-12-22 10:40:00 | 1695.60 | 2025-12-22 10:55:00 | 1705.50 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-12-22 10:40:00 | 1695.60 | 2025-12-22 11:20:00 | 1695.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-29 09:55:00 | 1654.90 | 2025-12-29 10:45:00 | 1646.48 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-12-29 09:55:00 | 1654.90 | 2025-12-29 13:50:00 | 1654.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-01 11:05:00 | 1710.90 | 2026-01-01 11:20:00 | 1705.56 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-01-01 11:05:00 | 1710.90 | 2026-01-01 12:30:00 | 1710.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-02 11:05:00 | 1711.00 | 2026-01-02 11:25:00 | 1705.03 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-01-02 11:05:00 | 1711.00 | 2026-01-02 15:10:00 | 1705.70 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2026-01-08 10:45:00 | 1669.30 | 2026-01-08 11:15:00 | 1663.39 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-01-08 10:45:00 | 1669.30 | 2026-01-08 11:25:00 | 1669.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-09 10:50:00 | 1622.20 | 2026-01-09 11:25:00 | 1616.18 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-01-09 10:50:00 | 1622.20 | 2026-01-09 14:55:00 | 1604.00 | TARGET_HIT | 0.50 | 1.12% |
| SELL | retest1 | 2026-01-14 10:00:00 | 1548.10 | 2026-01-14 11:25:00 | 1540.68 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-01-14 10:00:00 | 1548.10 | 2026-01-14 15:20:00 | 1525.00 | TARGET_HIT | 0.50 | 1.49% |
| SELL | retest1 | 2026-01-16 10:35:00 | 1522.90 | 2026-01-16 10:40:00 | 1517.88 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-01-16 10:35:00 | 1522.90 | 2026-01-16 12:05:00 | 1514.80 | TARGET_HIT | 0.50 | 0.53% |
| SELL | retest1 | 2026-01-23 09:55:00 | 1335.50 | 2026-01-23 11:35:00 | 1328.18 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-01-23 09:55:00 | 1335.50 | 2026-01-23 15:20:00 | 1303.30 | TARGET_HIT | 0.50 | 2.41% |
| SELL | retest1 | 2026-01-28 11:05:00 | 1285.50 | 2026-01-28 11:20:00 | 1289.94 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-01-30 09:30:00 | 1294.90 | 2026-01-30 09:45:00 | 1304.57 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2026-01-30 09:30:00 | 1294.90 | 2026-01-30 13:40:00 | 1306.60 | TARGET_HIT | 0.50 | 0.90% |
| BUY | retest1 | 2026-02-01 10:25:00 | 1340.30 | 2026-02-01 10:45:00 | 1347.27 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-02-01 10:25:00 | 1340.30 | 2026-02-01 11:30:00 | 1340.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-10 10:05:00 | 1295.40 | 2026-02-10 10:10:00 | 1302.75 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-02-10 10:05:00 | 1295.40 | 2026-02-10 11:30:00 | 1308.90 | TARGET_HIT | 0.50 | 1.04% |
| SELL | retest1 | 2026-02-23 10:55:00 | 1103.30 | 2026-02-23 11:00:00 | 1106.87 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-26 09:45:00 | 1075.80 | 2026-02-26 09:55:00 | 1070.44 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2026-03-04 10:45:00 | 1009.90 | 2026-03-04 11:30:00 | 1005.43 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-03-06 10:55:00 | 998.80 | 2026-03-06 13:55:00 | 1001.22 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-03-13 10:20:00 | 951.70 | 2026-03-13 10:50:00 | 956.23 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-03-20 11:15:00 | 885.30 | 2026-03-20 11:45:00 | 890.42 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-03-20 11:15:00 | 885.30 | 2026-03-20 11:50:00 | 885.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-25 10:00:00 | 893.80 | 2026-03-25 10:20:00 | 888.59 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2026-04-15 10:40:00 | 956.55 | 2026-04-15 10:50:00 | 951.90 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-04-16 09:30:00 | 939.60 | 2026-04-16 09:40:00 | 933.00 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2026-04-16 09:30:00 | 939.60 | 2026-04-16 14:45:00 | 922.55 | TARGET_HIT | 0.50 | 1.81% |
| BUY | retest1 | 2026-05-04 10:00:00 | 940.10 | 2026-05-04 10:10:00 | 935.86 | STOP_HIT | 1.00 | -0.45% |
