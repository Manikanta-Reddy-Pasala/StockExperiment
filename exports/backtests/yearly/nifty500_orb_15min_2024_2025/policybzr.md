# PB Fintech Ltd. (POLICYBZR)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-03-30 15:25:00 (34996 bars)
- **Last close:** 1424.10
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
| ENTRY1 | 22 |
| ENTRY2 | 0 |
| PARTIAL | 11 |
| TARGET_HIT | 6 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 33 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 16
- **Target hits / Stop hits / Partials:** 6 / 16 / 11
- **Avg / median % per leg:** 0.33% / 0.14%
- **Sum % (uncompounded):** 10.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 12 | 52.2% | 4 | 11 | 8 | 0.31% | 7.1% |
| BUY @ 2nd Alert (retest1) | 23 | 12 | 52.2% | 4 | 11 | 8 | 0.31% | 7.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 5 | 50.0% | 2 | 5 | 3 | 0.37% | 3.7% |
| SELL @ 2nd Alert (retest1) | 10 | 5 | 50.0% | 2 | 5 | 3 | 0.37% | 3.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 33 | 17 | 51.5% | 6 | 16 | 11 | 0.33% | 10.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-30 09:35:00 | 1191.10 | 1182.70 | 0.00 | ORB-long ORB[1174.40,1186.00] vol=1.5x ATR=3.56 |
| Stop hit — per-position SL triggered | 2024-05-30 09:40:00 | 1187.54 | 1183.11 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-06-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 11:05:00 | 1337.75 | 1355.60 | 0.00 | ORB-short ORB[1356.00,1366.30] vol=1.8x ATR=3.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-21 11:10:00 | 1332.58 | 1352.97 | 0.00 | T1 1.5R @ 1332.58 |
| Stop hit — per-position SL triggered | 2024-06-21 11:15:00 | 1337.75 | 1352.07 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-06-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:40:00 | 1358.80 | 1340.52 | 0.00 | ORB-long ORB[1330.00,1346.15] vol=1.9x ATR=5.33 |
| Stop hit — per-position SL triggered | 2024-06-26 10:45:00 | 1353.47 | 1340.87 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-07-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-30 09:35:00 | 1470.50 | 1480.02 | 0.00 | ORB-short ORB[1478.35,1497.35] vol=1.5x ATR=7.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 12:25:00 | 1459.76 | 1468.22 | 0.00 | T1 1.5R @ 1459.76 |
| Target hit | 2024-07-30 14:00:00 | 1464.90 | 1464.76 | 0.00 | Trail-exit close>VWAP |

### Cycle 5 — BUY (started 2024-09-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 09:35:00 | 1724.25 | 1721.71 | 0.00 | ORB-long ORB[1709.10,1722.40] vol=4.1x ATR=4.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 09:45:00 | 1730.34 | 1722.87 | 0.00 | T1 1.5R @ 1730.34 |
| Target hit | 2024-09-04 11:05:00 | 1726.60 | 1728.20 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2024-09-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 09:30:00 | 1762.00 | 1755.73 | 0.00 | ORB-long ORB[1738.55,1761.00] vol=1.7x ATR=7.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 10:10:00 | 1773.60 | 1760.63 | 0.00 | T1 1.5R @ 1773.60 |
| Target hit | 2024-09-11 15:20:00 | 1774.75 | 1771.62 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2024-10-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 09:55:00 | 1664.50 | 1652.31 | 0.00 | ORB-long ORB[1642.45,1654.00] vol=1.6x ATR=6.36 |
| Stop hit — per-position SL triggered | 2024-10-14 15:00:00 | 1658.14 | 1659.76 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-10-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:50:00 | 1630.50 | 1631.61 | 0.00 | ORB-short ORB[1633.50,1651.95] vol=9.1x ATR=8.42 |
| Stop hit — per-position SL triggered | 2024-10-25 09:55:00 | 1638.92 | 1631.81 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-11-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 11:10:00 | 1793.00 | 1818.22 | 0.00 | ORB-short ORB[1822.10,1844.40] vol=2.9x ATR=7.38 |
| Stop hit — per-position SL triggered | 2024-11-28 11:15:00 | 1800.38 | 1816.51 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-12-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 11:10:00 | 1991.30 | 1951.14 | 0.00 | ORB-long ORB[1914.95,1944.55] vol=2.5x ATR=8.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 11:25:00 | 2003.43 | 1962.72 | 0.00 | T1 1.5R @ 2003.43 |
| Stop hit — per-position SL triggered | 2024-12-04 11:55:00 | 1991.30 | 1970.38 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-12-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 09:30:00 | 2080.80 | 2053.60 | 0.00 | ORB-long ORB[2013.90,2032.65] vol=3.5x ATR=13.33 |
| Stop hit — per-position SL triggered | 2024-12-06 09:35:00 | 2067.47 | 2059.75 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-12-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 11:10:00 | 2153.35 | 2139.50 | 0.00 | ORB-long ORB[2101.50,2132.95] vol=4.1x ATR=7.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 11:35:00 | 2164.23 | 2142.53 | 0.00 | T1 1.5R @ 2164.23 |
| Stop hit — per-position SL triggered | 2024-12-17 12:20:00 | 2153.35 | 2145.83 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-12-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 11:00:00 | 2060.75 | 2054.00 | 0.00 | ORB-long ORB[2034.20,2054.45] vol=1.7x ATR=6.39 |
| Stop hit — per-position SL triggered | 2024-12-30 11:55:00 | 2054.36 | 2055.15 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-01-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:20:00 | 1699.60 | 1725.43 | 0.00 | ORB-short ORB[1731.45,1752.95] vol=1.7x ATR=7.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 10:35:00 | 1688.22 | 1721.58 | 0.00 | T1 1.5R @ 1688.22 |
| Target hit | 2025-01-21 15:20:00 | 1643.25 | 1675.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2025-01-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:30:00 | 1682.00 | 1648.16 | 0.00 | ORB-long ORB[1605.00,1629.00] vol=1.7x ATR=9.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 10:40:00 | 1695.84 | 1658.21 | 0.00 | T1 1.5R @ 1695.84 |
| Stop hit — per-position SL triggered | 2025-01-23 11:45:00 | 1682.00 | 1671.51 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-01-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-24 09:30:00 | 1702.75 | 1696.08 | 0.00 | ORB-long ORB[1682.45,1701.45] vol=2.2x ATR=6.84 |
| Stop hit — per-position SL triggered | 2025-01-24 09:45:00 | 1695.91 | 1697.70 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 11:15:00 | 1677.85 | 1669.68 | 0.00 | ORB-long ORB[1633.75,1653.00] vol=1.6x ATR=6.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 11:40:00 | 1687.11 | 1670.78 | 0.00 | T1 1.5R @ 1687.11 |
| Target hit | 2025-01-29 15:20:00 | 1717.40 | 1691.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — BUY (started 2025-02-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 09:40:00 | 1726.00 | 1722.73 | 0.00 | ORB-long ORB[1710.75,1724.60] vol=2.5x ATR=4.52 |
| Stop hit — per-position SL triggered | 2025-02-05 09:45:00 | 1721.48 | 1722.75 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-03-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 11:05:00 | 1443.50 | 1422.61 | 0.00 | ORB-long ORB[1405.30,1424.75] vol=2.1x ATR=6.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-11 11:20:00 | 1453.76 | 1427.19 | 0.00 | T1 1.5R @ 1453.76 |
| Target hit | 2025-03-11 15:20:00 | 1470.15 | 1448.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2025-04-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-21 09:55:00 | 1660.60 | 1664.98 | 0.00 | ORB-short ORB[1661.70,1686.00] vol=2.0x ATR=7.01 |
| Stop hit — per-position SL triggered | 2025-04-21 10:05:00 | 1667.61 | 1664.58 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-04-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:55:00 | 1638.80 | 1680.32 | 0.00 | ORB-short ORB[1690.10,1703.50] vol=1.9x ATR=7.35 |
| Stop hit — per-position SL triggered | 2025-04-23 11:05:00 | 1646.15 | 1672.59 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-04-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 10:50:00 | 1619.90 | 1609.58 | 0.00 | ORB-long ORB[1583.00,1607.10] vol=1.9x ATR=6.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 11:10:00 | 1629.02 | 1613.76 | 0.00 | T1 1.5R @ 1629.02 |
| Stop hit — per-position SL triggered | 2025-04-30 14:00:00 | 1619.90 | 1622.05 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-30 09:35:00 | 1191.10 | 2024-05-30 09:40:00 | 1187.54 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-06-21 11:05:00 | 1337.75 | 2024-06-21 11:10:00 | 1332.58 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-06-21 11:05:00 | 1337.75 | 2024-06-21 11:15:00 | 1337.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-26 10:40:00 | 1358.80 | 2024-06-26 10:45:00 | 1353.47 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-07-30 09:35:00 | 1470.50 | 2024-07-30 12:25:00 | 1459.76 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2024-07-30 09:35:00 | 1470.50 | 2024-07-30 14:00:00 | 1464.90 | TARGET_HIT | 0.50 | 0.38% |
| BUY | retest1 | 2024-09-04 09:35:00 | 1724.25 | 2024-09-04 09:45:00 | 1730.34 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-09-04 09:35:00 | 1724.25 | 2024-09-04 11:05:00 | 1726.60 | TARGET_HIT | 0.50 | 0.14% |
| BUY | retest1 | 2024-09-11 09:30:00 | 1762.00 | 2024-09-11 10:10:00 | 1773.60 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2024-09-11 09:30:00 | 1762.00 | 2024-09-11 15:20:00 | 1774.75 | TARGET_HIT | 0.50 | 0.72% |
| BUY | retest1 | 2024-10-14 09:55:00 | 1664.50 | 2024-10-14 15:00:00 | 1658.14 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-10-25 09:50:00 | 1630.50 | 2024-10-25 09:55:00 | 1638.92 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2024-11-28 11:10:00 | 1793.00 | 2024-11-28 11:15:00 | 1800.38 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-12-04 11:10:00 | 1991.30 | 2024-12-04 11:25:00 | 2003.43 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-12-04 11:10:00 | 1991.30 | 2024-12-04 11:55:00 | 1991.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-06 09:30:00 | 2080.80 | 2024-12-06 09:35:00 | 2067.47 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest1 | 2024-12-17 11:10:00 | 2153.35 | 2024-12-17 11:35:00 | 2164.23 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-12-17 11:10:00 | 2153.35 | 2024-12-17 12:20:00 | 2153.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-30 11:00:00 | 2060.75 | 2024-12-30 11:55:00 | 2054.36 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-01-21 10:20:00 | 1699.60 | 2025-01-21 10:35:00 | 1688.22 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2025-01-21 10:20:00 | 1699.60 | 2025-01-21 15:20:00 | 1643.25 | TARGET_HIT | 0.50 | 3.32% |
| BUY | retest1 | 2025-01-23 10:30:00 | 1682.00 | 2025-01-23 10:40:00 | 1695.84 | PARTIAL | 0.50 | 0.82% |
| BUY | retest1 | 2025-01-23 10:30:00 | 1682.00 | 2025-01-23 11:45:00 | 1682.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-24 09:30:00 | 1702.75 | 2025-01-24 09:45:00 | 1695.91 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-01-29 11:15:00 | 1677.85 | 2025-01-29 11:40:00 | 1687.11 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-01-29 11:15:00 | 1677.85 | 2025-01-29 15:20:00 | 1717.40 | TARGET_HIT | 0.50 | 2.36% |
| BUY | retest1 | 2025-02-05 09:40:00 | 1726.00 | 2025-02-05 09:45:00 | 1721.48 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-03-11 11:05:00 | 1443.50 | 2025-03-11 11:20:00 | 1453.76 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2025-03-11 11:05:00 | 1443.50 | 2025-03-11 15:20:00 | 1470.15 | TARGET_HIT | 0.50 | 1.85% |
| SELL | retest1 | 2025-04-21 09:55:00 | 1660.60 | 2025-04-21 10:05:00 | 1667.61 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-04-23 10:55:00 | 1638.80 | 2025-04-23 11:05:00 | 1646.15 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-04-30 10:50:00 | 1619.90 | 2025-04-30 11:10:00 | 1629.02 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-04-30 10:50:00 | 1619.90 | 2025-04-30 14:00:00 | 1619.90 | STOP_HIT | 0.50 | 0.00% |
