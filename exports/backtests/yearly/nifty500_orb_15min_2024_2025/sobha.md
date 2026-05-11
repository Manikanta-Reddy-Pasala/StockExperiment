# Sobha Ltd. (SOBHA)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
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
| ENTRY1 | 27 |
| ENTRY2 | 0 |
| PARTIAL | 11 |
| TARGET_HIT | 4 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 38 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 23
- **Target hits / Stop hits / Partials:** 4 / 23 / 11
- **Avg / median % per leg:** 0.20% / 0.00%
- **Sum % (uncompounded):** 7.43%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 4 | 33.3% | 1 | 8 | 3 | 0.01% | 0.1% |
| BUY @ 2nd Alert (retest1) | 12 | 4 | 33.3% | 1 | 8 | 3 | 0.01% | 0.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 26 | 11 | 42.3% | 3 | 15 | 8 | 0.28% | 7.4% |
| SELL @ 2nd Alert (retest1) | 26 | 11 | 42.3% | 3 | 15 | 8 | 0.28% | 7.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 38 | 15 | 39.5% | 4 | 23 | 11 | 0.20% | 7.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 10:05:00 | 2040.40 | 2026.38 | 0.00 | ORB-long ORB[2002.20,2031.35] vol=1.9x ATR=8.66 |
| Stop hit — per-position SL triggered | 2024-06-11 10:25:00 | 2031.74 | 2027.26 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-06-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 10:10:00 | 2029.45 | 2044.23 | 0.00 | ORB-short ORB[2045.00,2070.00] vol=2.7x ATR=9.66 |
| Stop hit — per-position SL triggered | 2024-06-27 10:15:00 | 2039.11 | 2041.29 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-07-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 09:30:00 | 2002.35 | 2005.45 | 0.00 | ORB-short ORB[2004.65,2027.00] vol=2.5x ATR=8.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 09:50:00 | 1989.73 | 2000.29 | 0.00 | T1 1.5R @ 1989.73 |
| Stop hit — per-position SL triggered | 2024-07-12 10:20:00 | 2002.35 | 1999.73 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-08-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-19 10:20:00 | 1706.80 | 1719.72 | 0.00 | ORB-short ORB[1718.00,1738.00] vol=2.2x ATR=6.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-19 11:00:00 | 1697.30 | 1715.12 | 0.00 | T1 1.5R @ 1697.30 |
| Stop hit — per-position SL triggered | 2024-08-19 11:25:00 | 1706.80 | 1713.97 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-08-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 10:00:00 | 1730.95 | 1719.72 | 0.00 | ORB-long ORB[1703.05,1725.80] vol=1.8x ATR=7.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 10:15:00 | 1742.90 | 1725.98 | 0.00 | T1 1.5R @ 1742.90 |
| Stop hit — per-position SL triggered | 2024-08-27 11:30:00 | 1730.95 | 1730.34 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 1690.10 | 1706.23 | 0.00 | ORB-short ORB[1706.70,1730.00] vol=3.0x ATR=8.79 |
| Stop hit — per-position SL triggered | 2024-08-28 09:35:00 | 1698.89 | 1704.77 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-08-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 09:35:00 | 1697.70 | 1706.05 | 0.00 | ORB-short ORB[1703.60,1716.80] vol=1.6x ATR=7.25 |
| Stop hit — per-position SL triggered | 2024-08-30 09:40:00 | 1704.95 | 1705.51 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-09-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 09:50:00 | 1742.95 | 1732.63 | 0.00 | ORB-long ORB[1717.00,1731.00] vol=1.9x ATR=5.90 |
| Stop hit — per-position SL triggered | 2024-09-04 10:30:00 | 1737.05 | 1735.17 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-09-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-11 10:45:00 | 1703.50 | 1717.17 | 0.00 | ORB-short ORB[1717.10,1733.40] vol=1.6x ATR=5.16 |
| Stop hit — per-position SL triggered | 2024-09-11 11:30:00 | 1708.66 | 1713.93 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-09-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 09:55:00 | 1752.50 | 1761.40 | 0.00 | ORB-short ORB[1753.55,1778.95] vol=1.7x ATR=6.47 |
| Stop hit — per-position SL triggered | 2024-09-17 10:00:00 | 1758.97 | 1760.57 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-09-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:20:00 | 1996.90 | 1992.91 | 0.00 | ORB-long ORB[1973.00,1994.05] vol=5.8x ATR=6.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 10:35:00 | 2006.57 | 1993.85 | 0.00 | T1 1.5R @ 2006.57 |
| Stop hit — per-position SL triggered | 2024-09-27 11:05:00 | 1996.90 | 1994.62 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-10-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 10:45:00 | 1904.20 | 1912.05 | 0.00 | ORB-short ORB[1917.00,1941.90] vol=2.2x ATR=7.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-01 11:10:00 | 1892.70 | 1909.12 | 0.00 | T1 1.5R @ 1892.70 |
| Target hit | 2024-10-01 15:20:00 | 1864.00 | 1892.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2024-10-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 11:05:00 | 1779.00 | 1794.14 | 0.00 | ORB-short ORB[1810.15,1825.00] vol=1.8x ATR=7.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:30:00 | 1767.87 | 1791.71 | 0.00 | T1 1.5R @ 1767.87 |
| Stop hit — per-position SL triggered | 2024-10-17 13:00:00 | 1779.00 | 1787.24 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-11-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 10:50:00 | 1593.10 | 1575.02 | 0.00 | ORB-long ORB[1530.05,1553.95] vol=2.0x ATR=9.71 |
| Stop hit — per-position SL triggered | 2024-11-19 10:55:00 | 1583.39 | 1575.60 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-11-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 09:30:00 | 1638.20 | 1629.39 | 0.00 | ORB-long ORB[1617.00,1636.55] vol=1.6x ATR=6.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-27 10:10:00 | 1648.32 | 1637.20 | 0.00 | T1 1.5R @ 1648.32 |
| Target hit | 2024-11-27 15:20:00 | 1652.05 | 1646.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2024-12-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 09:30:00 | 1661.65 | 1671.33 | 0.00 | ORB-short ORB[1666.80,1681.40] vol=1.5x ATR=6.16 |
| Stop hit — per-position SL triggered | 2024-12-06 09:35:00 | 1667.81 | 1671.14 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-12-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 10:25:00 | 1676.05 | 1664.32 | 0.00 | ORB-long ORB[1656.20,1671.00] vol=2.0x ATR=6.71 |
| Stop hit — per-position SL triggered | 2024-12-09 10:35:00 | 1669.34 | 1665.02 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-12-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-10 10:55:00 | 1644.10 | 1653.59 | 0.00 | ORB-short ORB[1655.10,1673.05] vol=2.4x ATR=5.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 11:35:00 | 1635.62 | 1650.54 | 0.00 | T1 1.5R @ 1635.62 |
| Stop hit — per-position SL triggered | 2024-12-10 12:30:00 | 1644.10 | 1646.33 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-12-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 10:10:00 | 1588.20 | 1610.24 | 0.00 | ORB-short ORB[1611.05,1629.95] vol=1.5x ATR=7.87 |
| Stop hit — per-position SL triggered | 2024-12-20 10:15:00 | 1596.07 | 1609.47 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-12-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 10:45:00 | 1586.00 | 1593.85 | 0.00 | ORB-short ORB[1586.40,1605.95] vol=2.7x ATR=3.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 11:20:00 | 1580.88 | 1592.63 | 0.00 | T1 1.5R @ 1580.88 |
| Stop hit — per-position SL triggered | 2024-12-27 13:35:00 | 1586.00 | 1584.48 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-01-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 10:05:00 | 1541.75 | 1546.84 | 0.00 | ORB-short ORB[1542.35,1563.70] vol=2.8x ATR=4.73 |
| Stop hit — per-position SL triggered | 2025-01-03 12:20:00 | 1546.48 | 1544.93 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 10:15:00 | 1339.10 | 1330.87 | 0.00 | ORB-long ORB[1320.65,1336.90] vol=1.9x ATR=5.47 |
| Stop hit — per-position SL triggered | 2025-01-16 10:25:00 | 1333.63 | 1331.83 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-01-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:00:00 | 1312.60 | 1319.68 | 0.00 | ORB-short ORB[1316.05,1333.60] vol=3.0x ATR=5.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 10:05:00 | 1304.53 | 1318.85 | 0.00 | T1 1.5R @ 1304.53 |
| Target hit | 2025-01-21 11:55:00 | 1308.15 | 1305.53 | 0.00 | Trail-exit close>VWAP |

### Cycle 24 — SELL (started 2025-02-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-05 10:00:00 | 1375.85 | 1386.94 | 0.00 | ORB-short ORB[1386.30,1403.70] vol=1.5x ATR=6.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-05 10:05:00 | 1365.82 | 1382.11 | 0.00 | T1 1.5R @ 1365.82 |
| Target hit | 2025-02-05 15:20:00 | 1315.20 | 1335.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — SELL (started 2025-02-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 09:40:00 | 1187.80 | 1192.75 | 0.00 | ORB-short ORB[1187.90,1203.00] vol=1.8x ATR=5.03 |
| Stop hit — per-position SL triggered | 2025-02-27 10:00:00 | 1192.83 | 1192.06 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-03-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 09:30:00 | 1264.60 | 1257.43 | 0.00 | ORB-long ORB[1243.35,1262.00] vol=1.8x ATR=4.94 |
| Stop hit — per-position SL triggered | 2025-03-20 09:35:00 | 1259.66 | 1257.27 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 09:35:00 | 1270.70 | 1278.51 | 0.00 | ORB-short ORB[1274.00,1286.00] vol=1.7x ATR=5.10 |
| Stop hit — per-position SL triggered | 2025-04-23 09:40:00 | 1275.80 | 1279.73 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-06-11 10:05:00 | 2040.40 | 2024-06-11 10:25:00 | 2031.74 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-06-27 10:10:00 | 2029.45 | 2024-06-27 10:15:00 | 2039.11 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-07-12 09:30:00 | 2002.35 | 2024-07-12 09:50:00 | 1989.73 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-07-12 09:30:00 | 2002.35 | 2024-07-12 10:20:00 | 2002.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-19 10:20:00 | 1706.80 | 2024-08-19 11:00:00 | 1697.30 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-08-19 10:20:00 | 1706.80 | 2024-08-19 11:25:00 | 1706.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-27 10:00:00 | 1730.95 | 2024-08-27 10:15:00 | 1742.90 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-08-27 10:00:00 | 1730.95 | 2024-08-27 11:30:00 | 1730.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-28 09:30:00 | 1690.10 | 2024-08-28 09:35:00 | 1698.89 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2024-08-30 09:35:00 | 1697.70 | 2024-08-30 09:40:00 | 1704.95 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-09-04 09:50:00 | 1742.95 | 2024-09-04 10:30:00 | 1737.05 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-09-11 10:45:00 | 1703.50 | 2024-09-11 11:30:00 | 1708.66 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-09-17 09:55:00 | 1752.50 | 2024-09-17 10:00:00 | 1758.97 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-09-27 10:20:00 | 1996.90 | 2024-09-27 10:35:00 | 2006.57 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-09-27 10:20:00 | 1996.90 | 2024-09-27 11:05:00 | 1996.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-01 10:45:00 | 1904.20 | 2024-10-01 11:10:00 | 1892.70 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-10-01 10:45:00 | 1904.20 | 2024-10-01 15:20:00 | 1864.00 | TARGET_HIT | 0.50 | 2.11% |
| SELL | retest1 | 2024-10-17 11:05:00 | 1779.00 | 2024-10-17 11:30:00 | 1767.87 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-10-17 11:05:00 | 1779.00 | 2024-10-17 13:00:00 | 1779.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-19 10:50:00 | 1593.10 | 2024-11-19 10:55:00 | 1583.39 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest1 | 2024-11-27 09:30:00 | 1638.20 | 2024-11-27 10:10:00 | 1648.32 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-11-27 09:30:00 | 1638.20 | 2024-11-27 15:20:00 | 1652.05 | TARGET_HIT | 0.50 | 0.85% |
| SELL | retest1 | 2024-12-06 09:30:00 | 1661.65 | 2024-12-06 09:35:00 | 1667.81 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-12-09 10:25:00 | 1676.05 | 2024-12-09 10:35:00 | 1669.34 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-12-10 10:55:00 | 1644.10 | 2024-12-10 11:35:00 | 1635.62 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-12-10 10:55:00 | 1644.10 | 2024-12-10 12:30:00 | 1644.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-20 10:10:00 | 1588.20 | 2024-12-20 10:15:00 | 1596.07 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-12-27 10:45:00 | 1586.00 | 2024-12-27 11:20:00 | 1580.88 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-12-27 10:45:00 | 1586.00 | 2024-12-27 13:35:00 | 1586.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-03 10:05:00 | 1541.75 | 2025-01-03 12:20:00 | 1546.48 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-01-16 10:15:00 | 1339.10 | 2025-01-16 10:25:00 | 1333.63 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-01-21 10:00:00 | 1312.60 | 2025-01-21 10:05:00 | 1304.53 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2025-01-21 10:00:00 | 1312.60 | 2025-01-21 11:55:00 | 1308.15 | TARGET_HIT | 0.50 | 0.34% |
| SELL | retest1 | 2025-02-05 10:00:00 | 1375.85 | 2025-02-05 10:05:00 | 1365.82 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2025-02-05 10:00:00 | 1375.85 | 2025-02-05 15:20:00 | 1315.20 | TARGET_HIT | 0.50 | 4.41% |
| SELL | retest1 | 2025-02-27 09:40:00 | 1187.80 | 2025-02-27 10:00:00 | 1192.83 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-03-20 09:30:00 | 1264.60 | 2025-03-20 09:35:00 | 1259.66 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-04-23 09:35:00 | 1270.70 | 2025-04-23 09:40:00 | 1275.80 | STOP_HIT | 1.00 | -0.40% |
