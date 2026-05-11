# Prestige Estates Projects Ltd. (PRESTIGE)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
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
| ENTRY1 | 25 |
| ENTRY2 | 0 |
| PARTIAL | 13 |
| TARGET_HIT | 6 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 38 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 19
- **Target hits / Stop hits / Partials:** 6 / 19 / 13
- **Avg / median % per leg:** 0.34% / 0.15%
- **Sum % (uncompounded):** 13.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 6 | 50.0% | 2 | 6 | 4 | 0.15% | 1.8% |
| BUY @ 2nd Alert (retest1) | 12 | 6 | 50.0% | 2 | 6 | 4 | 0.15% | 1.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 26 | 13 | 50.0% | 4 | 13 | 9 | 0.43% | 11.2% |
| SELL @ 2nd Alert (retest1) | 26 | 13 | 50.0% | 4 | 13 | 9 | 0.43% | 11.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 38 | 19 | 50.0% | 6 | 19 | 13 | 0.34% | 13.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 10:35:00 | 1546.35 | 1529.30 | 0.00 | ORB-long ORB[1520.05,1539.55] vol=2.2x ATR=6.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 10:45:00 | 1556.78 | 1533.94 | 0.00 | T1 1.5R @ 1556.78 |
| Target hit | 2024-05-17 15:00:00 | 1558.70 | 1562.67 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — SELL (started 2024-07-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 11:05:00 | 1764.85 | 1788.15 | 0.00 | ORB-short ORB[1788.00,1804.70] vol=2.1x ATR=9.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 12:20:00 | 1749.92 | 1780.44 | 0.00 | T1 1.5R @ 1749.92 |
| Stop hit — per-position SL triggered | 2024-07-10 12:35:00 | 1764.85 | 1777.64 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-07-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 10:10:00 | 1775.70 | 1782.21 | 0.00 | ORB-short ORB[1780.25,1800.00] vol=5.4x ATR=10.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 10:35:00 | 1760.21 | 1776.13 | 0.00 | T1 1.5R @ 1760.21 |
| Stop hit — per-position SL triggered | 2024-07-11 14:20:00 | 1775.70 | 1770.43 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-09-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-11 09:35:00 | 1803.10 | 1807.22 | 0.00 | ORB-short ORB[1805.25,1830.00] vol=1.9x ATR=10.45 |
| Stop hit — per-position SL triggered | 2024-09-11 09:45:00 | 1813.55 | 1808.71 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-09-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 10:20:00 | 1835.00 | 1819.05 | 0.00 | ORB-long ORB[1794.00,1817.80] vol=3.5x ATR=8.76 |
| Stop hit — per-position SL triggered | 2024-09-12 10:25:00 | 1826.24 | 1822.08 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-09-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 11:05:00 | 1900.40 | 1883.20 | 0.00 | ORB-long ORB[1871.00,1898.80] vol=4.6x ATR=7.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 12:00:00 | 1911.42 | 1891.93 | 0.00 | T1 1.5R @ 1911.42 |
| Stop hit — per-position SL triggered | 2024-09-23 12:20:00 | 1900.40 | 1893.19 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-09-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-30 10:10:00 | 1823.05 | 1836.75 | 0.00 | ORB-short ORB[1831.25,1858.40] vol=2.4x ATR=10.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 11:45:00 | 1808.03 | 1830.42 | 0.00 | T1 1.5R @ 1808.03 |
| Stop hit — per-position SL triggered | 2024-09-30 14:45:00 | 1823.05 | 1820.67 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-10-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 11:10:00 | 1799.20 | 1812.81 | 0.00 | ORB-short ORB[1819.55,1832.50] vol=2.2x ATR=6.52 |
| Stop hit — per-position SL triggered | 2024-10-01 11:45:00 | 1805.72 | 1808.39 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-10-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:40:00 | 1709.95 | 1719.59 | 0.00 | ORB-short ORB[1720.10,1741.30] vol=1.8x ATR=9.14 |
| Stop hit — per-position SL triggered | 2024-10-07 11:20:00 | 1719.09 | 1717.72 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-10-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-28 11:00:00 | 1666.95 | 1674.33 | 0.00 | ORB-short ORB[1671.75,1695.65] vol=2.0x ATR=7.99 |
| Stop hit — per-position SL triggered | 2024-10-28 12:00:00 | 1674.94 | 1669.95 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-10-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 10:40:00 | 1610.50 | 1647.64 | 0.00 | ORB-short ORB[1652.00,1673.05] vol=1.9x ATR=7.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 11:10:00 | 1599.91 | 1629.21 | 0.00 | T1 1.5R @ 1599.91 |
| Stop hit — per-position SL triggered | 2024-10-29 12:35:00 | 1610.50 | 1616.80 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-11-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 10:00:00 | 1561.00 | 1566.69 | 0.00 | ORB-short ORB[1571.00,1590.55] vol=2.7x ATR=8.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 10:10:00 | 1547.72 | 1562.92 | 0.00 | T1 1.5R @ 1547.72 |
| Target hit | 2024-11-13 15:20:00 | 1529.95 | 1528.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2024-11-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-29 10:55:00 | 1652.75 | 1655.53 | 0.00 | ORB-short ORB[1656.05,1668.90] vol=4.2x ATR=5.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 11:40:00 | 1645.10 | 1653.27 | 0.00 | T1 1.5R @ 1645.10 |
| Stop hit — per-position SL triggered | 2024-11-29 12:40:00 | 1652.75 | 1650.70 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-12-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 11:05:00 | 1737.65 | 1745.14 | 0.00 | ORB-short ORB[1741.05,1760.40] vol=5.2x ATR=5.29 |
| Stop hit — per-position SL triggered | 2024-12-27 11:10:00 | 1742.94 | 1744.89 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-01-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-01 10:50:00 | 1674.20 | 1679.51 | 0.00 | ORB-short ORB[1675.80,1693.95] vol=2.2x ATR=6.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 11:35:00 | 1665.18 | 1676.64 | 0.00 | T1 1.5R @ 1665.18 |
| Target hit | 2025-01-01 15:20:00 | 1655.10 | 1660.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2025-01-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 10:50:00 | 1648.60 | 1664.04 | 0.00 | ORB-short ORB[1662.20,1682.00] vol=1.7x ATR=4.79 |
| Stop hit — per-position SL triggered | 2025-01-03 11:00:00 | 1653.39 | 1663.43 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-01-06 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 10:25:00 | 1637.90 | 1650.36 | 0.00 | ORB-short ORB[1650.10,1670.00] vol=2.6x ATR=6.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 10:45:00 | 1628.47 | 1647.24 | 0.00 | T1 1.5R @ 1628.47 |
| Target hit | 2025-01-06 12:30:00 | 1635.40 | 1630.40 | 0.00 | Trail-exit close>VWAP |

### Cycle 18 — SELL (started 2025-01-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 09:35:00 | 1442.65 | 1454.93 | 0.00 | ORB-short ORB[1450.10,1470.00] vol=1.6x ATR=8.99 |
| Stop hit — per-position SL triggered | 2025-01-13 10:05:00 | 1451.64 | 1452.17 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-01-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 11:05:00 | 1321.85 | 1338.11 | 0.00 | ORB-short ORB[1340.05,1358.65] vol=2.2x ATR=5.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 11:15:00 | 1312.89 | 1329.70 | 0.00 | T1 1.5R @ 1312.89 |
| Target hit | 2025-01-24 15:20:00 | 1253.65 | 1275.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 10:15:00 | 1287.55 | 1282.27 | 0.00 | ORB-long ORB[1261.25,1277.95] vol=1.6x ATR=6.19 |
| Stop hit — per-position SL triggered | 2025-01-29 10:25:00 | 1281.36 | 1282.59 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-01-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 10:40:00 | 1336.40 | 1334.97 | 0.00 | ORB-long ORB[1312.20,1328.30] vol=1.5x ATR=6.64 |
| Stop hit — per-position SL triggered | 2025-01-30 10:55:00 | 1329.76 | 1334.75 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 11:15:00 | 1356.55 | 1369.68 | 0.00 | ORB-short ORB[1375.00,1391.45] vol=1.7x ATR=4.95 |
| Stop hit — per-position SL triggered | 2025-02-06 11:20:00 | 1361.50 | 1369.46 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-03-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:40:00 | 1115.10 | 1105.10 | 0.00 | ORB-long ORB[1094.90,1109.75] vol=1.9x ATR=5.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 10:15:00 | 1122.77 | 1109.95 | 0.00 | T1 1.5R @ 1122.77 |
| Target hit | 2025-03-18 11:55:00 | 1119.85 | 1120.89 | 0.00 | Trail-exit close<VWAP |

### Cycle 24 — BUY (started 2025-04-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 11:10:00 | 1230.90 | 1225.27 | 0.00 | ORB-long ORB[1207.40,1225.50] vol=1.6x ATR=4.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 11:30:00 | 1237.39 | 1225.87 | 0.00 | T1 1.5R @ 1237.39 |
| Stop hit — per-position SL triggered | 2025-04-21 12:00:00 | 1230.90 | 1227.53 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-05-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 10:50:00 | 1389.70 | 1379.35 | 0.00 | ORB-long ORB[1360.10,1378.10] vol=1.6x ATR=5.70 |
| Stop hit — per-position SL triggered | 2025-05-05 11:10:00 | 1384.00 | 1380.00 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-17 10:35:00 | 1546.35 | 2024-05-17 10:45:00 | 1556.78 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2024-05-17 10:35:00 | 1546.35 | 2024-05-17 15:00:00 | 1558.70 | TARGET_HIT | 0.50 | 0.80% |
| SELL | retest1 | 2024-07-10 11:05:00 | 1764.85 | 2024-07-10 12:20:00 | 1749.92 | PARTIAL | 0.50 | 0.85% |
| SELL | retest1 | 2024-07-10 11:05:00 | 1764.85 | 2024-07-10 12:35:00 | 1764.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-11 10:10:00 | 1775.70 | 2024-07-11 10:35:00 | 1760.21 | PARTIAL | 0.50 | 0.87% |
| SELL | retest1 | 2024-07-11 10:10:00 | 1775.70 | 2024-07-11 14:20:00 | 1775.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-11 09:35:00 | 1803.10 | 2024-09-11 09:45:00 | 1813.55 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2024-09-12 10:20:00 | 1835.00 | 2024-09-12 10:25:00 | 1826.24 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-09-23 11:05:00 | 1900.40 | 2024-09-23 12:00:00 | 1911.42 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-09-23 11:05:00 | 1900.40 | 2024-09-23 12:20:00 | 1900.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-30 10:10:00 | 1823.05 | 2024-09-30 11:45:00 | 1808.03 | PARTIAL | 0.50 | 0.82% |
| SELL | retest1 | 2024-09-30 10:10:00 | 1823.05 | 2024-09-30 14:45:00 | 1823.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-01 11:10:00 | 1799.20 | 2024-10-01 11:45:00 | 1805.72 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-10-07 10:40:00 | 1709.95 | 2024-10-07 11:20:00 | 1719.09 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2024-10-28 11:00:00 | 1666.95 | 2024-10-28 12:00:00 | 1674.94 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-10-29 10:40:00 | 1610.50 | 2024-10-29 11:10:00 | 1599.91 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-10-29 10:40:00 | 1610.50 | 2024-10-29 12:35:00 | 1610.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-13 10:00:00 | 1561.00 | 2024-11-13 10:10:00 | 1547.72 | PARTIAL | 0.50 | 0.85% |
| SELL | retest1 | 2024-11-13 10:00:00 | 1561.00 | 2024-11-13 15:20:00 | 1529.95 | TARGET_HIT | 0.50 | 1.99% |
| SELL | retest1 | 2024-11-29 10:55:00 | 1652.75 | 2024-11-29 11:40:00 | 1645.10 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-11-29 10:55:00 | 1652.75 | 2024-11-29 12:40:00 | 1652.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-27 11:05:00 | 1737.65 | 2024-12-27 11:10:00 | 1742.94 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-01-01 10:50:00 | 1674.20 | 2025-01-01 11:35:00 | 1665.18 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-01-01 10:50:00 | 1674.20 | 2025-01-01 15:20:00 | 1655.10 | TARGET_HIT | 0.50 | 1.14% |
| SELL | retest1 | 2025-01-03 10:50:00 | 1648.60 | 2025-01-03 11:00:00 | 1653.39 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-01-06 10:25:00 | 1637.90 | 2025-01-06 10:45:00 | 1628.47 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-01-06 10:25:00 | 1637.90 | 2025-01-06 12:30:00 | 1635.40 | TARGET_HIT | 0.50 | 0.15% |
| SELL | retest1 | 2025-01-13 09:35:00 | 1442.65 | 2025-01-13 10:05:00 | 1451.64 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest1 | 2025-01-24 11:05:00 | 1321.85 | 2025-01-24 11:15:00 | 1312.89 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2025-01-24 11:05:00 | 1321.85 | 2025-01-24 15:20:00 | 1253.65 | TARGET_HIT | 0.50 | 5.16% |
| BUY | retest1 | 2025-01-29 10:15:00 | 1287.55 | 2025-01-29 10:25:00 | 1281.36 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-01-30 10:40:00 | 1336.40 | 2025-01-30 10:55:00 | 1329.76 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2025-02-06 11:15:00 | 1356.55 | 2025-02-06 11:20:00 | 1361.50 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-03-18 09:40:00 | 1115.10 | 2025-03-18 10:15:00 | 1122.77 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2025-03-18 09:40:00 | 1115.10 | 2025-03-18 11:55:00 | 1119.85 | TARGET_HIT | 0.50 | 0.43% |
| BUY | retest1 | 2025-04-21 11:10:00 | 1230.90 | 2025-04-21 11:30:00 | 1237.39 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-04-21 11:10:00 | 1230.90 | 2025-04-21 12:00:00 | 1230.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-05 10:50:00 | 1389.70 | 2025-05-05 11:10:00 | 1384.00 | STOP_HIT | 1.00 | -0.41% |
