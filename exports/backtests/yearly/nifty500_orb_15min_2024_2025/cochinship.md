# Cochin Shipyard Ltd. (COCHINSHIP)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 1769.40
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
| ENTRY1 | 18 |
| ENTRY2 | 0 |
| PARTIAL | 9 |
| TARGET_HIT | 3 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 15
- **Target hits / Stop hits / Partials:** 3 / 15 / 9
- **Avg / median % per leg:** 0.19% / 0.00%
- **Sum % (uncompounded):** 5.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 5 | 38.5% | 2 | 8 | 3 | 0.09% | 1.1% |
| BUY @ 2nd Alert (retest1) | 13 | 5 | 38.5% | 2 | 8 | 3 | 0.09% | 1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 7 | 50.0% | 1 | 7 | 6 | 0.28% | 3.9% |
| SELL @ 2nd Alert (retest1) | 14 | 7 | 50.0% | 1 | 7 | 6 | 0.28% | 3.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 27 | 12 | 44.4% | 3 | 15 | 9 | 0.19% | 5.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 09:30:00 | 1919.30 | 1905.69 | 0.00 | ORB-long ORB[1887.00,1914.50] vol=4.1x ATR=8.86 |
| Stop hit — per-position SL triggered | 2024-06-12 10:15:00 | 1910.44 | 1909.73 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-06-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 11:00:00 | 1900.00 | 1909.08 | 0.00 | ORB-short ORB[1905.80,1923.30] vol=1.6x ATR=4.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 11:15:00 | 1893.51 | 1907.97 | 0.00 | T1 1.5R @ 1893.51 |
| Stop hit — per-position SL triggered | 2024-06-13 11:45:00 | 1900.00 | 1906.65 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-06-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 09:45:00 | 2172.90 | 2160.53 | 0.00 | ORB-long ORB[2145.15,2170.00] vol=3.2x ATR=9.81 |
| Stop hit — per-position SL triggered | 2024-06-25 09:50:00 | 2163.09 | 2161.03 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-07-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-25 09:35:00 | 2494.40 | 2514.23 | 0.00 | ORB-short ORB[2504.15,2531.75] vol=2.7x ATR=13.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 09:50:00 | 2473.56 | 2505.94 | 0.00 | T1 1.5R @ 2473.56 |
| Stop hit — per-position SL triggered | 2024-07-25 10:40:00 | 2494.40 | 2492.34 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-09-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-27 09:30:00 | 1696.00 | 1708.32 | 0.00 | ORB-short ORB[1700.00,1724.65] vol=2.4x ATR=8.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 09:45:00 | 1683.27 | 1701.99 | 0.00 | T1 1.5R @ 1683.27 |
| Target hit | 2024-09-27 10:25:00 | 1694.00 | 1693.71 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — BUY (started 2024-10-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 11:05:00 | 1648.00 | 1632.29 | 0.00 | ORB-long ORB[1624.40,1640.00] vol=3.6x ATR=8.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-15 11:10:00 | 1661.03 | 1637.93 | 0.00 | T1 1.5R @ 1661.03 |
| Target hit | 2024-10-15 15:20:00 | 1673.00 | 1657.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2024-10-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-17 09:30:00 | 1566.00 | 1560.91 | 0.00 | ORB-long ORB[1556.10,1565.00] vol=2.4x ATR=4.50 |
| Stop hit — per-position SL triggered | 2024-10-17 09:40:00 | 1561.50 | 1561.49 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-10-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:50:00 | 1526.10 | 1554.55 | 0.00 | ORB-short ORB[1550.00,1570.00] vol=1.8x ATR=8.21 |
| Stop hit — per-position SL triggered | 2024-10-21 09:55:00 | 1534.31 | 1552.85 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-11-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:30:00 | 1341.50 | 1353.97 | 0.00 | ORB-short ORB[1350.00,1365.05] vol=2.4x ATR=10.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:35:00 | 1325.37 | 1341.48 | 0.00 | T1 1.5R @ 1325.37 |
| Stop hit — per-position SL triggered | 2024-11-13 09:55:00 | 1341.50 | 1334.11 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-11-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 10:35:00 | 1362.30 | 1339.25 | 0.00 | ORB-long ORB[1320.00,1337.60] vol=3.5x ATR=6.34 |
| Stop hit — per-position SL triggered | 2024-11-19 15:15:00 | 1355.96 | 1350.56 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-12-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 11:00:00 | 1612.80 | 1619.91 | 0.00 | ORB-short ORB[1615.00,1638.40] vol=2.6x ATR=8.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 13:25:00 | 1599.51 | 1614.03 | 0.00 | T1 1.5R @ 1599.51 |
| Stop hit — per-position SL triggered | 2024-12-16 13:30:00 | 1612.80 | 1613.84 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-12-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 09:45:00 | 1598.40 | 1603.88 | 0.00 | ORB-short ORB[1602.65,1614.00] vol=3.6x ATR=8.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 09:50:00 | 1586.05 | 1602.91 | 0.00 | T1 1.5R @ 1586.05 |
| Stop hit — per-position SL triggered | 2024-12-17 10:05:00 | 1598.40 | 1600.28 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-01-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 10:00:00 | 1595.00 | 1578.08 | 0.00 | ORB-long ORB[1552.65,1575.00] vol=3.6x ATR=8.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 10:05:00 | 1607.88 | 1590.03 | 0.00 | T1 1.5R @ 1607.88 |
| Stop hit — per-position SL triggered | 2025-01-02 10:35:00 | 1595.00 | 1594.38 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-01-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 09:30:00 | 1586.95 | 1596.59 | 0.00 | ORB-short ORB[1595.25,1607.80] vol=2.2x ATR=6.20 |
| Stop hit — per-position SL triggered | 2025-01-03 09:55:00 | 1593.15 | 1593.62 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-04-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 09:30:00 | 1441.10 | 1433.68 | 0.00 | ORB-long ORB[1416.30,1437.70] vol=4.1x ATR=6.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-15 09:55:00 | 1451.25 | 1439.41 | 0.00 | T1 1.5R @ 1451.25 |
| Target hit | 2025-04-15 11:25:00 | 1451.20 | 1455.07 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — BUY (started 2025-04-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 09:55:00 | 1471.20 | 1448.53 | 0.00 | ORB-long ORB[1431.00,1449.10] vol=2.9x ATR=7.25 |
| Stop hit — per-position SL triggered | 2025-04-16 10:00:00 | 1463.95 | 1451.97 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-04-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:55:00 | 1557.40 | 1515.41 | 0.00 | ORB-long ORB[1455.10,1478.00] vol=4.7x ATR=11.37 |
| Stop hit — per-position SL triggered | 2025-04-21 10:05:00 | 1546.03 | 1526.52 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 09:30:00 | 1510.20 | 1502.95 | 0.00 | ORB-long ORB[1483.10,1498.10] vol=7.8x ATR=7.35 |
| Stop hit — per-position SL triggered | 2025-04-24 09:35:00 | 1502.85 | 1503.23 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-06-12 09:30:00 | 1919.30 | 2024-06-12 10:15:00 | 1910.44 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-06-13 11:00:00 | 1900.00 | 2024-06-13 11:15:00 | 1893.51 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-06-13 11:00:00 | 1900.00 | 2024-06-13 11:45:00 | 1900.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-25 09:45:00 | 2172.90 | 2024-06-25 09:50:00 | 2163.09 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-07-25 09:35:00 | 2494.40 | 2024-07-25 09:50:00 | 2473.56 | PARTIAL | 0.50 | 0.84% |
| SELL | retest1 | 2024-07-25 09:35:00 | 2494.40 | 2024-07-25 10:40:00 | 2494.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-27 09:30:00 | 1696.00 | 2024-09-27 09:45:00 | 1683.27 | PARTIAL | 0.50 | 0.75% |
| SELL | retest1 | 2024-09-27 09:30:00 | 1696.00 | 2024-09-27 10:25:00 | 1694.00 | TARGET_HIT | 0.50 | 0.12% |
| BUY | retest1 | 2024-10-15 11:05:00 | 1648.00 | 2024-10-15 11:10:00 | 1661.03 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2024-10-15 11:05:00 | 1648.00 | 2024-10-15 15:20:00 | 1673.00 | TARGET_HIT | 0.50 | 1.52% |
| BUY | retest1 | 2024-10-17 09:30:00 | 1566.00 | 2024-10-17 09:40:00 | 1561.50 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-10-21 09:50:00 | 1526.10 | 2024-10-21 09:55:00 | 1534.31 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2024-11-13 09:30:00 | 1341.50 | 2024-11-13 09:35:00 | 1325.37 | PARTIAL | 0.50 | 1.20% |
| SELL | retest1 | 2024-11-13 09:30:00 | 1341.50 | 2024-11-13 09:55:00 | 1341.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-19 10:35:00 | 1362.30 | 2024-11-19 15:15:00 | 1355.96 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-12-16 11:00:00 | 1612.80 | 2024-12-16 13:25:00 | 1599.51 | PARTIAL | 0.50 | 0.82% |
| SELL | retest1 | 2024-12-16 11:00:00 | 1612.80 | 2024-12-16 13:30:00 | 1612.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-17 09:45:00 | 1598.40 | 2024-12-17 09:50:00 | 1586.05 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2024-12-17 09:45:00 | 1598.40 | 2024-12-17 10:05:00 | 1598.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-02 10:00:00 | 1595.00 | 2025-01-02 10:05:00 | 1607.88 | PARTIAL | 0.50 | 0.81% |
| BUY | retest1 | 2025-01-02 10:00:00 | 1595.00 | 2025-01-02 10:35:00 | 1595.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-03 09:30:00 | 1586.95 | 2025-01-03 09:55:00 | 1593.15 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-04-15 09:30:00 | 1441.10 | 2025-04-15 09:55:00 | 1451.25 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2025-04-15 09:30:00 | 1441.10 | 2025-04-15 11:25:00 | 1451.20 | TARGET_HIT | 0.50 | 0.70% |
| BUY | retest1 | 2025-04-16 09:55:00 | 1471.20 | 2025-04-16 10:00:00 | 1463.95 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-04-21 09:55:00 | 1557.40 | 2025-04-21 10:05:00 | 1546.03 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest1 | 2025-04-24 09:30:00 | 1510.20 | 2025-04-24 09:35:00 | 1502.85 | STOP_HIT | 1.00 | -0.49% |
