# Coromandel International Ltd. (COROMANDEL)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1928.90
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
| ENTRY1 | 75 |
| ENTRY2 | 0 |
| PARTIAL | 37 |
| TARGET_HIT | 18 |
| STOP_HIT | 57 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 112 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 55 / 57
- **Target hits / Stop hits / Partials:** 18 / 57 / 37
- **Avg / median % per leg:** 0.25% / 0.00%
- **Sum % (uncompounded):** 28.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 59 | 28 | 47.5% | 8 | 31 | 20 | 0.31% | 18.3% |
| BUY @ 2nd Alert (retest1) | 59 | 28 | 47.5% | 8 | 31 | 20 | 0.31% | 18.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 53 | 27 | 50.9% | 10 | 26 | 17 | 0.19% | 10.2% |
| SELL @ 2nd Alert (retest1) | 53 | 27 | 50.9% | 10 | 26 | 17 | 0.19% | 10.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 112 | 55 | 49.1% | 18 | 57 | 37 | 0.25% | 28.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-13 11:15:00 | 1192.70 | 1184.45 | 0.00 | ORB-long ORB[1178.00,1190.00] vol=3.4x ATR=5.22 |
| Stop hit — per-position SL triggered | 2024-05-13 11:30:00 | 1187.48 | 1184.94 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 11:00:00 | 1220.00 | 1208.87 | 0.00 | ORB-long ORB[1201.75,1214.95] vol=2.6x ATR=3.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-14 11:15:00 | 1225.47 | 1212.15 | 0.00 | T1 1.5R @ 1225.47 |
| Target hit | 2024-05-14 15:20:00 | 1233.05 | 1226.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2024-05-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 09:45:00 | 1253.25 | 1248.72 | 0.00 | ORB-long ORB[1245.20,1251.00] vol=1.6x ATR=3.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 09:50:00 | 1258.08 | 1252.45 | 0.00 | T1 1.5R @ 1258.08 |
| Stop hit — per-position SL triggered | 2024-05-17 10:05:00 | 1253.25 | 1253.15 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-22 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 10:20:00 | 1227.65 | 1230.67 | 0.00 | ORB-short ORB[1230.05,1242.35] vol=1.7x ATR=3.73 |
| Stop hit — per-position SL triggered | 2024-05-22 10:30:00 | 1231.38 | 1230.51 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 11:10:00 | 1232.80 | 1236.40 | 0.00 | ORB-short ORB[1233.45,1249.00] vol=2.6x ATR=3.13 |
| Stop hit — per-position SL triggered | 2024-05-23 12:35:00 | 1235.93 | 1235.01 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-05-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 10:30:00 | 1249.70 | 1235.36 | 0.00 | ORB-long ORB[1225.40,1232.50] vol=2.0x ATR=3.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-24 10:40:00 | 1255.64 | 1245.76 | 0.00 | T1 1.5R @ 1255.64 |
| Target hit | 2024-05-24 14:25:00 | 1255.45 | 1257.43 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — SELL (started 2024-05-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 10:55:00 | 1246.05 | 1250.66 | 0.00 | ORB-short ORB[1251.25,1264.00] vol=1.6x ATR=4.15 |
| Stop hit — per-position SL triggered | 2024-05-27 13:55:00 | 1250.20 | 1248.69 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-07 09:35:00 | 1391.95 | 1399.57 | 0.00 | ORB-short ORB[1392.95,1410.00] vol=2.4x ATR=6.47 |
| Stop hit — per-position SL triggered | 2024-06-07 10:15:00 | 1398.42 | 1396.81 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 10:50:00 | 1489.80 | 1484.07 | 0.00 | ORB-long ORB[1473.15,1485.00] vol=1.6x ATR=4.43 |
| Stop hit — per-position SL triggered | 2024-06-13 11:05:00 | 1485.37 | 1484.50 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 11:15:00 | 1500.55 | 1495.65 | 0.00 | ORB-long ORB[1482.10,1495.00] vol=2.1x ATR=2.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 12:50:00 | 1504.91 | 1497.92 | 0.00 | T1 1.5R @ 1504.91 |
| Stop hit — per-position SL triggered | 2024-06-14 13:05:00 | 1500.55 | 1498.78 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 10:10:00 | 1588.00 | 1573.53 | 0.00 | ORB-long ORB[1554.30,1577.30] vol=1.6x ATR=7.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-20 10:50:00 | 1598.89 | 1586.96 | 0.00 | T1 1.5R @ 1598.89 |
| Target hit | 2024-06-20 15:15:00 | 1640.65 | 1642.51 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — BUY (started 2024-06-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 09:40:00 | 1553.75 | 1543.17 | 0.00 | ORB-long ORB[1530.05,1547.00] vol=1.5x ATR=5.62 |
| Stop hit — per-position SL triggered | 2024-06-25 09:45:00 | 1548.13 | 1543.76 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-06-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:00:00 | 1538.25 | 1531.60 | 0.00 | ORB-long ORB[1521.00,1533.35] vol=4.1x ATR=5.93 |
| Stop hit — per-position SL triggered | 2024-06-26 10:05:00 | 1532.32 | 1531.73 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-02 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 10:35:00 | 1580.05 | 1590.50 | 0.00 | ORB-short ORB[1593.55,1605.05] vol=1.9x ATR=4.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 10:55:00 | 1573.89 | 1587.89 | 0.00 | T1 1.5R @ 1573.89 |
| Target hit | 2024-07-02 15:20:00 | 1569.85 | 1576.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — SELL (started 2024-07-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 09:30:00 | 1575.25 | 1577.83 | 0.00 | ORB-short ORB[1576.05,1592.00] vol=1.6x ATR=4.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 10:50:00 | 1568.23 | 1575.11 | 0.00 | T1 1.5R @ 1568.23 |
| Target hit | 2024-07-04 11:10:00 | 1575.00 | 1574.81 | 0.00 | Trail-exit close>VWAP |

### Cycle 16 — BUY (started 2024-07-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 10:05:00 | 1608.90 | 1596.93 | 0.00 | ORB-long ORB[1590.15,1607.95] vol=3.0x ATR=6.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-08 10:15:00 | 1618.07 | 1601.40 | 0.00 | T1 1.5R @ 1618.07 |
| Target hit | 2024-07-08 11:20:00 | 1618.95 | 1620.09 | 0.00 | Trail-exit close<VWAP |

### Cycle 17 — SELL (started 2024-07-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:30:00 | 1584.60 | 1588.69 | 0.00 | ORB-short ORB[1597.65,1617.05] vol=2.0x ATR=5.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:35:00 | 1575.69 | 1588.35 | 0.00 | T1 1.5R @ 1575.69 |
| Stop hit — per-position SL triggered | 2024-07-10 10:40:00 | 1584.60 | 1587.94 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 09:30:00 | 1629.55 | 1613.56 | 0.00 | ORB-long ORB[1597.45,1614.00] vol=3.2x ATR=5.75 |
| Stop hit — per-position SL triggered | 2024-07-12 09:35:00 | 1623.80 | 1615.72 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 10:15:00 | 1614.15 | 1603.56 | 0.00 | ORB-long ORB[1590.35,1609.00] vol=2.2x ATR=5.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 10:55:00 | 1622.31 | 1609.33 | 0.00 | T1 1.5R @ 1622.31 |
| Stop hit — per-position SL triggered | 2024-07-15 11:00:00 | 1614.15 | 1609.19 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-07-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-16 11:05:00 | 1593.50 | 1600.47 | 0.00 | ORB-short ORB[1595.00,1606.40] vol=3.8x ATR=4.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 12:10:00 | 1587.35 | 1596.97 | 0.00 | T1 1.5R @ 1587.35 |
| Target hit | 2024-07-16 15:20:00 | 1583.90 | 1591.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — SELL (started 2024-07-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 10:05:00 | 1578.60 | 1589.54 | 0.00 | ORB-short ORB[1585.20,1601.05] vol=1.6x ATR=5.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 10:10:00 | 1570.24 | 1588.91 | 0.00 | T1 1.5R @ 1570.24 |
| Stop hit — per-position SL triggered | 2024-07-19 11:40:00 | 1578.60 | 1580.98 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-07-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-25 09:45:00 | 1598.95 | 1601.66 | 0.00 | ORB-short ORB[1605.60,1620.00] vol=8.8x ATR=6.54 |
| Stop hit — per-position SL triggered | 2024-07-25 10:10:00 | 1605.49 | 1601.74 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 09:30:00 | 1707.75 | 1713.32 | 0.00 | ORB-short ORB[1714.00,1730.65] vol=2.9x ATR=6.23 |
| Stop hit — per-position SL triggered | 2024-08-14 09:45:00 | 1713.98 | 1710.77 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 11:05:00 | 1718.85 | 1729.75 | 0.00 | ORB-short ORB[1725.10,1750.00] vol=3.5x ATR=4.75 |
| Stop hit — per-position SL triggered | 2024-08-20 11:15:00 | 1723.60 | 1729.54 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:45:00 | 1756.50 | 1745.33 | 0.00 | ORB-long ORB[1737.10,1750.50] vol=4.5x ATR=5.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 10:10:00 | 1765.32 | 1747.74 | 0.00 | T1 1.5R @ 1765.32 |
| Stop hit — per-position SL triggered | 2024-08-21 10:30:00 | 1756.50 | 1749.49 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-08-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 10:30:00 | 1769.25 | 1772.33 | 0.00 | ORB-short ORB[1770.00,1780.50] vol=3.1x ATR=4.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 11:45:00 | 1762.90 | 1770.00 | 0.00 | T1 1.5R @ 1762.90 |
| Target hit | 2024-08-23 15:20:00 | 1749.00 | 1763.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2024-08-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 09:55:00 | 1764.95 | 1760.64 | 0.00 | ORB-long ORB[1746.15,1762.30] vol=2.0x ATR=4.76 |
| Stop hit — per-position SL triggered | 2024-08-26 10:00:00 | 1760.19 | 1760.13 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 1730.55 | 1741.22 | 0.00 | ORB-short ORB[1738.80,1754.75] vol=2.9x ATR=5.06 |
| Stop hit — per-position SL triggered | 2024-08-28 09:35:00 | 1735.61 | 1738.99 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-08-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:55:00 | 1714.90 | 1727.17 | 0.00 | ORB-short ORB[1725.30,1741.40] vol=1.7x ATR=3.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 11:45:00 | 1709.05 | 1722.12 | 0.00 | T1 1.5R @ 1709.05 |
| Stop hit — per-position SL triggered | 2024-08-29 12:20:00 | 1714.90 | 1718.13 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 09:55:00 | 1730.25 | 1724.25 | 0.00 | ORB-long ORB[1705.50,1727.95] vol=1.6x ATR=5.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 10:00:00 | 1739.15 | 1726.48 | 0.00 | T1 1.5R @ 1739.15 |
| Stop hit — per-position SL triggered | 2024-09-04 10:20:00 | 1730.25 | 1727.22 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-06 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:25:00 | 1698.50 | 1699.26 | 0.00 | ORB-short ORB[1703.15,1725.00] vol=2.3x ATR=5.94 |
| Stop hit — per-position SL triggered | 2024-09-06 12:25:00 | 1704.44 | 1698.34 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-09-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-13 11:15:00 | 1689.80 | 1697.15 | 0.00 | ORB-short ORB[1690.20,1706.25] vol=1.7x ATR=3.63 |
| Stop hit — per-position SL triggered | 2024-09-13 11:35:00 | 1693.43 | 1696.03 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 10:40:00 | 1726.35 | 1718.75 | 0.00 | ORB-long ORB[1705.30,1725.90] vol=2.1x ATR=4.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 10:55:00 | 1733.61 | 1722.82 | 0.00 | T1 1.5R @ 1733.61 |
| Stop hit — per-position SL triggered | 2024-09-17 11:40:00 | 1726.35 | 1729.13 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 10:15:00 | 1709.45 | 1712.19 | 0.00 | ORB-short ORB[1712.50,1725.00] vol=6.4x ATR=4.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 10:45:00 | 1702.00 | 1709.94 | 0.00 | T1 1.5R @ 1702.00 |
| Stop hit — per-position SL triggered | 2024-09-18 11:45:00 | 1709.45 | 1709.25 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-09-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 11:05:00 | 1689.15 | 1716.59 | 0.00 | ORB-short ORB[1726.00,1741.95] vol=1.9x ATR=6.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 11:45:00 | 1680.10 | 1711.73 | 0.00 | T1 1.5R @ 1680.10 |
| Target hit | 2024-09-19 15:20:00 | 1676.50 | 1697.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — SELL (started 2024-09-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-30 09:30:00 | 1649.55 | 1656.14 | 0.00 | ORB-short ORB[1652.30,1669.25] vol=2.1x ATR=4.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 09:45:00 | 1642.09 | 1654.14 | 0.00 | T1 1.5R @ 1642.09 |
| Stop hit — per-position SL triggered | 2024-09-30 09:50:00 | 1649.55 | 1653.52 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-10-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 10:30:00 | 1695.95 | 1688.65 | 0.00 | ORB-long ORB[1675.00,1693.90] vol=1.6x ATR=7.03 |
| Stop hit — per-position SL triggered | 2024-10-01 10:45:00 | 1688.92 | 1688.99 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-10-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 09:45:00 | 1585.65 | 1578.93 | 0.00 | ORB-long ORB[1566.65,1582.40] vol=1.6x ATR=6.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-09 09:55:00 | 1594.86 | 1584.85 | 0.00 | T1 1.5R @ 1594.86 |
| Target hit | 2024-10-09 10:55:00 | 1604.10 | 1604.23 | 0.00 | Trail-exit close<VWAP |

### Cycle 39 — SELL (started 2024-10-15 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-15 10:30:00 | 1618.90 | 1627.00 | 0.00 | ORB-short ORB[1625.05,1639.10] vol=2.7x ATR=4.22 |
| Stop hit — per-position SL triggered | 2024-10-15 11:00:00 | 1623.12 | 1626.33 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-10-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 11:00:00 | 1607.85 | 1613.81 | 0.00 | ORB-short ORB[1612.70,1628.95] vol=2.6x ATR=4.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:40:00 | 1601.26 | 1611.00 | 0.00 | T1 1.5R @ 1601.26 |
| Target hit | 2024-10-17 15:20:00 | 1591.10 | 1603.62 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — BUY (started 2024-10-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-18 09:50:00 | 1593.45 | 1585.77 | 0.00 | ORB-long ORB[1570.70,1583.95] vol=1.9x ATR=6.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 10:35:00 | 1603.50 | 1588.89 | 0.00 | T1 1.5R @ 1603.50 |
| Target hit | 2024-10-18 15:20:00 | 1619.20 | 1608.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — SELL (started 2024-10-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:40:00 | 1604.05 | 1608.71 | 0.00 | ORB-short ORB[1605.00,1628.95] vol=2.0x ATR=5.50 |
| Stop hit — per-position SL triggered | 2024-10-21 09:55:00 | 1609.55 | 1606.99 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-10-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:35:00 | 1590.30 | 1601.10 | 0.00 | ORB-short ORB[1590.70,1610.85] vol=1.7x ATR=5.84 |
| Stop hit — per-position SL triggered | 2024-10-22 11:05:00 | 1596.14 | 1599.01 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-10-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 09:45:00 | 1599.00 | 1606.15 | 0.00 | ORB-short ORB[1601.15,1613.55] vol=1.6x ATR=5.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 10:15:00 | 1590.09 | 1600.20 | 0.00 | T1 1.5R @ 1590.09 |
| Target hit | 2024-10-29 14:05:00 | 1589.25 | 1588.91 | 0.00 | Trail-exit close>VWAP |

### Cycle 45 — BUY (started 2024-10-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 09:35:00 | 1627.15 | 1617.25 | 0.00 | ORB-long ORB[1601.65,1621.00] vol=1.8x ATR=5.99 |
| Stop hit — per-position SL triggered | 2024-10-30 09:45:00 | 1621.16 | 1618.25 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 11:15:00 | 1685.85 | 1683.65 | 0.00 | ORB-long ORB[1668.10,1684.00] vol=2.4x ATR=3.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 11:20:00 | 1691.55 | 1684.01 | 0.00 | T1 1.5R @ 1691.55 |
| Target hit | 2024-11-06 15:20:00 | 1729.65 | 1712.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — BUY (started 2024-11-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-07 10:20:00 | 1758.70 | 1744.13 | 0.00 | ORB-long ORB[1729.10,1745.85] vol=2.9x ATR=6.57 |
| Stop hit — per-position SL triggered | 2024-11-07 10:25:00 | 1752.13 | 1746.10 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-11-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 10:20:00 | 1767.00 | 1750.96 | 0.00 | ORB-long ORB[1724.55,1735.25] vol=5.6x ATR=5.20 |
| Stop hit — per-position SL triggered | 2024-11-19 10:25:00 | 1761.80 | 1752.00 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-11-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:40:00 | 1771.05 | 1763.82 | 0.00 | ORB-long ORB[1748.05,1763.45] vol=1.7x ATR=4.18 |
| Stop hit — per-position SL triggered | 2024-11-28 10:15:00 | 1766.87 | 1766.30 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-11-29 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 10:30:00 | 1774.40 | 1764.50 | 0.00 | ORB-long ORB[1755.85,1769.75] vol=1.5x ATR=5.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 10:45:00 | 1782.89 | 1768.66 | 0.00 | T1 1.5R @ 1782.89 |
| Stop hit — per-position SL triggered | 2024-11-29 12:45:00 | 1774.40 | 1774.06 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-12-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-03 10:05:00 | 1767.90 | 1776.74 | 0.00 | ORB-short ORB[1768.60,1785.00] vol=3.3x ATR=5.48 |
| Stop hit — per-position SL triggered | 2024-12-03 10:35:00 | 1773.38 | 1774.75 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-12-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:40:00 | 1802.35 | 1794.88 | 0.00 | ORB-long ORB[1774.25,1799.00] vol=2.4x ATR=4.98 |
| Stop hit — per-position SL triggered | 2024-12-04 09:55:00 | 1797.37 | 1799.34 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-12-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 10:50:00 | 1795.75 | 1787.23 | 0.00 | ORB-long ORB[1766.10,1791.95] vol=1.9x ATR=5.47 |
| Stop hit — per-position SL triggered | 2024-12-10 11:00:00 | 1790.28 | 1787.35 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-12-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 09:40:00 | 1780.45 | 1775.25 | 0.00 | ORB-long ORB[1762.20,1779.45] vol=2.8x ATR=5.05 |
| Stop hit — per-position SL triggered | 2024-12-12 09:50:00 | 1775.40 | 1775.94 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-12-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 09:30:00 | 1764.75 | 1768.46 | 0.00 | ORB-short ORB[1770.00,1783.75] vol=12.2x ATR=6.16 |
| Stop hit — per-position SL triggered | 2024-12-16 09:45:00 | 1770.91 | 1767.94 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-12-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 09:50:00 | 1834.65 | 1825.57 | 0.00 | ORB-long ORB[1811.70,1833.40] vol=1.6x ATR=6.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 12:00:00 | 1843.90 | 1833.39 | 0.00 | T1 1.5R @ 1843.90 |
| Stop hit — per-position SL triggered | 2024-12-17 12:25:00 | 1834.65 | 1835.02 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2024-12-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-23 11:15:00 | 1853.20 | 1834.88 | 0.00 | ORB-long ORB[1819.15,1845.35] vol=3.7x ATR=6.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 11:40:00 | 1862.77 | 1844.86 | 0.00 | T1 1.5R @ 1862.77 |
| Stop hit — per-position SL triggered | 2024-12-23 12:35:00 | 1853.20 | 1852.65 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-12-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 11:05:00 | 1836.00 | 1843.14 | 0.00 | ORB-short ORB[1842.25,1861.40] vol=2.4x ATR=5.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 11:25:00 | 1828.44 | 1841.15 | 0.00 | T1 1.5R @ 1828.44 |
| Target hit | 2024-12-26 14:15:00 | 1835.20 | 1829.99 | 0.00 | Trail-exit close>VWAP |

### Cycle 59 — SELL (started 2025-01-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 11:00:00 | 1906.00 | 1912.16 | 0.00 | ORB-short ORB[1914.30,1928.50] vol=2.2x ATR=3.90 |
| Stop hit — per-position SL triggered | 2025-01-02 11:05:00 | 1909.90 | 1911.71 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-01-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 11:05:00 | 1947.00 | 1963.09 | 0.00 | ORB-short ORB[1949.05,1970.30] vol=1.7x ATR=6.09 |
| Stop hit — per-position SL triggered | 2025-01-06 11:30:00 | 1953.09 | 1961.20 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-01-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 10:45:00 | 1962.90 | 1960.17 | 0.00 | ORB-long ORB[1944.05,1957.50] vol=2.5x ATR=5.99 |
| Stop hit — per-position SL triggered | 2025-01-07 11:15:00 | 1956.91 | 1960.51 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-01-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 09:35:00 | 1953.05 | 1947.89 | 0.00 | ORB-long ORB[1927.50,1951.45] vol=1.9x ATR=5.86 |
| Stop hit — per-position SL triggered | 2025-01-09 09:40:00 | 1947.19 | 1948.53 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-01-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:50:00 | 1918.25 | 1925.35 | 0.00 | ORB-short ORB[1930.85,1955.00] vol=2.3x ATR=6.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:55:00 | 1908.37 | 1922.92 | 0.00 | T1 1.5R @ 1908.37 |
| Stop hit — per-position SL triggered | 2025-01-10 10:05:00 | 1918.25 | 1921.23 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-01-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-22 09:35:00 | 1835.25 | 1843.70 | 0.00 | ORB-short ORB[1838.50,1854.25] vol=1.9x ATR=5.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:50:00 | 1827.27 | 1840.78 | 0.00 | T1 1.5R @ 1827.27 |
| Target hit | 2025-01-22 15:20:00 | 1780.65 | 1787.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 65 — BUY (started 2025-01-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 09:45:00 | 1730.35 | 1721.51 | 0.00 | ORB-long ORB[1705.00,1724.25] vol=2.0x ATR=7.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 10:05:00 | 1741.24 | 1725.00 | 0.00 | T1 1.5R @ 1741.24 |
| Target hit | 2025-01-29 15:20:00 | 1797.25 | 1763.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 66 — BUY (started 2025-01-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:40:00 | 1829.35 | 1806.61 | 0.00 | ORB-long ORB[1791.25,1805.00] vol=2.1x ATR=8.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 09:50:00 | 1841.51 | 1818.83 | 0.00 | T1 1.5R @ 1841.51 |
| Stop hit — per-position SL triggered | 2025-01-30 10:20:00 | 1829.35 | 1825.40 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-02-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 09:45:00 | 1902.20 | 1893.42 | 0.00 | ORB-long ORB[1863.00,1888.65] vol=2.0x ATR=7.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-05 10:00:00 | 1913.07 | 1898.09 | 0.00 | T1 1.5R @ 1913.07 |
| Stop hit — per-position SL triggered | 2025-02-05 10:20:00 | 1902.20 | 1899.94 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-02-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 11:00:00 | 1871.30 | 1885.31 | 0.00 | ORB-short ORB[1881.05,1899.90] vol=1.5x ATR=6.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 11:05:00 | 1861.33 | 1883.94 | 0.00 | T1 1.5R @ 1861.33 |
| Stop hit — per-position SL triggered | 2025-02-06 11:10:00 | 1871.30 | 1883.57 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-02-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-11 09:45:00 | 1834.20 | 1839.01 | 0.00 | ORB-short ORB[1834.45,1855.45] vol=1.8x ATR=5.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 10:10:00 | 1825.98 | 1836.44 | 0.00 | T1 1.5R @ 1825.98 |
| Target hit | 2025-02-11 14:45:00 | 1817.70 | 1817.40 | 0.00 | Trail-exit close>VWAP |

### Cycle 70 — SELL (started 2025-02-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-18 10:50:00 | 1669.05 | 1686.76 | 0.00 | ORB-short ORB[1705.90,1725.40] vol=1.5x ATR=6.98 |
| Stop hit — per-position SL triggered | 2025-02-18 11:00:00 | 1676.03 | 1684.99 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-03-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 10:40:00 | 2010.40 | 1996.52 | 0.00 | ORB-long ORB[1979.10,2007.95] vol=2.1x ATR=9.32 |
| Stop hit — per-position SL triggered | 2025-03-26 12:50:00 | 2001.08 | 2008.40 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-03-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 10:45:00 | 2014.90 | 1999.70 | 0.00 | ORB-long ORB[1960.00,1989.95] vol=2.5x ATR=7.78 |
| Stop hit — per-position SL triggered | 2025-03-27 10:50:00 | 2007.12 | 1999.39 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2025-04-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 10:10:00 | 2181.60 | 2176.68 | 0.00 | ORB-long ORB[2157.50,2178.60] vol=6.7x ATR=7.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 10:25:00 | 2193.54 | 2178.03 | 0.00 | T1 1.5R @ 2193.54 |
| Stop hit — per-position SL triggered | 2025-04-23 10:55:00 | 2181.60 | 2179.87 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 10:15:00 | 2194.60 | 2207.43 | 0.00 | ORB-short ORB[2222.60,2250.90] vol=4.1x ATR=9.87 |
| Stop hit — per-position SL triggered | 2025-04-25 12:15:00 | 2204.47 | 2200.08 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2025-04-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-29 09:35:00 | 2291.50 | 2259.89 | 0.00 | ORB-long ORB[2239.90,2252.60] vol=2.9x ATR=13.29 |
| Stop hit — per-position SL triggered | 2025-04-29 09:40:00 | 2278.21 | 2262.72 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-13 11:15:00 | 1192.70 | 2024-05-13 11:30:00 | 1187.48 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-05-14 11:00:00 | 1220.00 | 2024-05-14 11:15:00 | 1225.47 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-05-14 11:00:00 | 1220.00 | 2024-05-14 15:20:00 | 1233.05 | TARGET_HIT | 0.50 | 1.07% |
| BUY | retest1 | 2024-05-17 09:45:00 | 1253.25 | 2024-05-17 09:50:00 | 1258.08 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-05-17 09:45:00 | 1253.25 | 2024-05-17 10:05:00 | 1253.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-22 10:20:00 | 1227.65 | 2024-05-22 10:30:00 | 1231.38 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-05-23 11:10:00 | 1232.80 | 2024-05-23 12:35:00 | 1235.93 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-05-24 10:30:00 | 1249.70 | 2024-05-24 10:40:00 | 1255.64 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-05-24 10:30:00 | 1249.70 | 2024-05-24 14:25:00 | 1255.45 | TARGET_HIT | 0.50 | 0.46% |
| SELL | retest1 | 2024-05-27 10:55:00 | 1246.05 | 2024-05-27 13:55:00 | 1250.20 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-06-07 09:35:00 | 1391.95 | 2024-06-07 10:15:00 | 1398.42 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-06-13 10:50:00 | 1489.80 | 2024-06-13 11:05:00 | 1485.37 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-06-14 11:15:00 | 1500.55 | 2024-06-14 12:50:00 | 1504.91 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2024-06-14 11:15:00 | 1500.55 | 2024-06-14 13:05:00 | 1500.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-20 10:10:00 | 1588.00 | 2024-06-20 10:50:00 | 1598.89 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-06-20 10:10:00 | 1588.00 | 2024-06-20 15:15:00 | 1640.65 | TARGET_HIT | 0.50 | 3.32% |
| BUY | retest1 | 2024-06-25 09:40:00 | 1553.75 | 2024-06-25 09:45:00 | 1548.13 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-06-26 10:00:00 | 1538.25 | 2024-06-26 10:05:00 | 1532.32 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-07-02 10:35:00 | 1580.05 | 2024-07-02 10:55:00 | 1573.89 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-07-02 10:35:00 | 1580.05 | 2024-07-02 15:20:00 | 1569.85 | TARGET_HIT | 0.50 | 0.65% |
| SELL | retest1 | 2024-07-04 09:30:00 | 1575.25 | 2024-07-04 10:50:00 | 1568.23 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-07-04 09:30:00 | 1575.25 | 2024-07-04 11:10:00 | 1575.00 | TARGET_HIT | 0.50 | 0.02% |
| BUY | retest1 | 2024-07-08 10:05:00 | 1608.90 | 2024-07-08 10:15:00 | 1618.07 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-07-08 10:05:00 | 1608.90 | 2024-07-08 11:20:00 | 1618.95 | TARGET_HIT | 0.50 | 0.62% |
| SELL | retest1 | 2024-07-10 10:30:00 | 1584.60 | 2024-07-10 10:35:00 | 1575.69 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-07-10 10:30:00 | 1584.60 | 2024-07-10 10:40:00 | 1584.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-12 09:30:00 | 1629.55 | 2024-07-12 09:35:00 | 1623.80 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-07-15 10:15:00 | 1614.15 | 2024-07-15 10:55:00 | 1622.31 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-07-15 10:15:00 | 1614.15 | 2024-07-15 11:00:00 | 1614.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-16 11:05:00 | 1593.50 | 2024-07-16 12:10:00 | 1587.35 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-07-16 11:05:00 | 1593.50 | 2024-07-16 15:20:00 | 1583.90 | TARGET_HIT | 0.50 | 0.60% |
| SELL | retest1 | 2024-07-19 10:05:00 | 1578.60 | 2024-07-19 10:10:00 | 1570.24 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-07-19 10:05:00 | 1578.60 | 2024-07-19 11:40:00 | 1578.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-25 09:45:00 | 1598.95 | 2024-07-25 10:10:00 | 1605.49 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-08-14 09:30:00 | 1707.75 | 2024-08-14 09:45:00 | 1713.98 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-08-20 11:05:00 | 1718.85 | 2024-08-20 11:15:00 | 1723.60 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-08-21 09:45:00 | 1756.50 | 2024-08-21 10:10:00 | 1765.32 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-08-21 09:45:00 | 1756.50 | 2024-08-21 10:30:00 | 1756.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-23 10:30:00 | 1769.25 | 2024-08-23 11:45:00 | 1762.90 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-08-23 10:30:00 | 1769.25 | 2024-08-23 15:20:00 | 1749.00 | TARGET_HIT | 0.50 | 1.14% |
| BUY | retest1 | 2024-08-26 09:55:00 | 1764.95 | 2024-08-26 10:00:00 | 1760.19 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-08-28 09:30:00 | 1730.55 | 2024-08-28 09:35:00 | 1735.61 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-08-29 10:55:00 | 1714.90 | 2024-08-29 11:45:00 | 1709.05 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-08-29 10:55:00 | 1714.90 | 2024-08-29 12:20:00 | 1714.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-04 09:55:00 | 1730.25 | 2024-09-04 10:00:00 | 1739.15 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-09-04 09:55:00 | 1730.25 | 2024-09-04 10:20:00 | 1730.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-06 10:25:00 | 1698.50 | 2024-09-06 12:25:00 | 1704.44 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-09-13 11:15:00 | 1689.80 | 2024-09-13 11:35:00 | 1693.43 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-09-17 10:40:00 | 1726.35 | 2024-09-17 10:55:00 | 1733.61 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-09-17 10:40:00 | 1726.35 | 2024-09-17 11:40:00 | 1726.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-18 10:15:00 | 1709.45 | 2024-09-18 10:45:00 | 1702.00 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-09-18 10:15:00 | 1709.45 | 2024-09-18 11:45:00 | 1709.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-19 11:05:00 | 1689.15 | 2024-09-19 11:45:00 | 1680.10 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-09-19 11:05:00 | 1689.15 | 2024-09-19 15:20:00 | 1676.50 | TARGET_HIT | 0.50 | 0.75% |
| SELL | retest1 | 2024-09-30 09:30:00 | 1649.55 | 2024-09-30 09:45:00 | 1642.09 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-09-30 09:30:00 | 1649.55 | 2024-09-30 09:50:00 | 1649.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-01 10:30:00 | 1695.95 | 2024-10-01 10:45:00 | 1688.92 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-10-09 09:45:00 | 1585.65 | 2024-10-09 09:55:00 | 1594.86 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-10-09 09:45:00 | 1585.65 | 2024-10-09 10:55:00 | 1604.10 | TARGET_HIT | 0.50 | 1.16% |
| SELL | retest1 | 2024-10-15 10:30:00 | 1618.90 | 2024-10-15 11:00:00 | 1623.12 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-10-17 11:00:00 | 1607.85 | 2024-10-17 11:40:00 | 1601.26 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-10-17 11:00:00 | 1607.85 | 2024-10-17 15:20:00 | 1591.10 | TARGET_HIT | 0.50 | 1.04% |
| BUY | retest1 | 2024-10-18 09:50:00 | 1593.45 | 2024-10-18 10:35:00 | 1603.50 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-10-18 09:50:00 | 1593.45 | 2024-10-18 15:20:00 | 1619.20 | TARGET_HIT | 0.50 | 1.62% |
| SELL | retest1 | 2024-10-21 09:40:00 | 1604.05 | 2024-10-21 09:55:00 | 1609.55 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-10-22 10:35:00 | 1590.30 | 2024-10-22 11:05:00 | 1596.14 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-10-29 09:45:00 | 1599.00 | 2024-10-29 10:15:00 | 1590.09 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-10-29 09:45:00 | 1599.00 | 2024-10-29 14:05:00 | 1589.25 | TARGET_HIT | 0.50 | 0.61% |
| BUY | retest1 | 2024-10-30 09:35:00 | 1627.15 | 2024-10-30 09:45:00 | 1621.16 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-11-06 11:15:00 | 1685.85 | 2024-11-06 11:20:00 | 1691.55 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-11-06 11:15:00 | 1685.85 | 2024-11-06 15:20:00 | 1729.65 | TARGET_HIT | 0.50 | 2.60% |
| BUY | retest1 | 2024-11-07 10:20:00 | 1758.70 | 2024-11-07 10:25:00 | 1752.13 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-11-19 10:20:00 | 1767.00 | 2024-11-19 10:25:00 | 1761.80 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-11-28 09:40:00 | 1771.05 | 2024-11-28 10:15:00 | 1766.87 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-11-29 10:30:00 | 1774.40 | 2024-11-29 10:45:00 | 1782.89 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-11-29 10:30:00 | 1774.40 | 2024-11-29 12:45:00 | 1774.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-03 10:05:00 | 1767.90 | 2024-12-03 10:35:00 | 1773.38 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-12-04 09:40:00 | 1802.35 | 2024-12-04 09:55:00 | 1797.37 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-12-10 10:50:00 | 1795.75 | 2024-12-10 11:00:00 | 1790.28 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-12-12 09:40:00 | 1780.45 | 2024-12-12 09:50:00 | 1775.40 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-12-16 09:30:00 | 1764.75 | 2024-12-16 09:45:00 | 1770.91 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-12-17 09:50:00 | 1834.65 | 2024-12-17 12:00:00 | 1843.90 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-12-17 09:50:00 | 1834.65 | 2024-12-17 12:25:00 | 1834.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-23 11:15:00 | 1853.20 | 2024-12-23 11:40:00 | 1862.77 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-12-23 11:15:00 | 1853.20 | 2024-12-23 12:35:00 | 1853.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-26 11:05:00 | 1836.00 | 2024-12-26 11:25:00 | 1828.44 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-12-26 11:05:00 | 1836.00 | 2024-12-26 14:15:00 | 1835.20 | TARGET_HIT | 0.50 | 0.04% |
| SELL | retest1 | 2025-01-02 11:00:00 | 1906.00 | 2025-01-02 11:05:00 | 1909.90 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-01-06 11:05:00 | 1947.00 | 2025-01-06 11:30:00 | 1953.09 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-01-07 10:45:00 | 1962.90 | 2025-01-07 11:15:00 | 1956.91 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-01-09 09:35:00 | 1953.05 | 2025-01-09 09:40:00 | 1947.19 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-01-10 09:50:00 | 1918.25 | 2025-01-10 09:55:00 | 1908.37 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-01-10 09:50:00 | 1918.25 | 2025-01-10 10:05:00 | 1918.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-22 09:35:00 | 1835.25 | 2025-01-22 09:50:00 | 1827.27 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-01-22 09:35:00 | 1835.25 | 2025-01-22 15:20:00 | 1780.65 | TARGET_HIT | 0.50 | 2.98% |
| BUY | retest1 | 2025-01-29 09:45:00 | 1730.35 | 2025-01-29 10:05:00 | 1741.24 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-01-29 09:45:00 | 1730.35 | 2025-01-29 15:20:00 | 1797.25 | TARGET_HIT | 0.50 | 3.87% |
| BUY | retest1 | 2025-01-30 09:40:00 | 1829.35 | 2025-01-30 09:50:00 | 1841.51 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-01-30 09:40:00 | 1829.35 | 2025-01-30 10:20:00 | 1829.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-05 09:45:00 | 1902.20 | 2025-02-05 10:00:00 | 1913.07 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-02-05 09:45:00 | 1902.20 | 2025-02-05 10:20:00 | 1902.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-06 11:00:00 | 1871.30 | 2025-02-06 11:05:00 | 1861.33 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-02-06 11:00:00 | 1871.30 | 2025-02-06 11:10:00 | 1871.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-11 09:45:00 | 1834.20 | 2025-02-11 10:10:00 | 1825.98 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-02-11 09:45:00 | 1834.20 | 2025-02-11 14:45:00 | 1817.70 | TARGET_HIT | 0.50 | 0.90% |
| SELL | retest1 | 2025-02-18 10:50:00 | 1669.05 | 2025-02-18 11:00:00 | 1676.03 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-03-26 10:40:00 | 2010.40 | 2025-03-26 12:50:00 | 2001.08 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-03-27 10:45:00 | 2014.90 | 2025-03-27 10:50:00 | 2007.12 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-04-23 10:10:00 | 2181.60 | 2025-04-23 10:25:00 | 2193.54 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-04-23 10:10:00 | 2181.60 | 2025-04-23 10:55:00 | 2181.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-25 10:15:00 | 2194.60 | 2025-04-25 12:15:00 | 2204.47 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-04-29 09:35:00 | 2291.50 | 2025-04-29 09:40:00 | 2278.21 | STOP_HIT | 1.00 | -0.58% |
