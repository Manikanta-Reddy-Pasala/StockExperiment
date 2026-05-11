# Garden Reach Shipbuilders & Engineers Ltd. (GRSE)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-08-01 15:25:00 (22908 bars)
- **Last close:** 2580.30
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
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 16
- **Target hits / Stop hits / Partials:** 2 / 16 / 3
- **Avg / median % per leg:** 0.02% / -0.40%
- **Sum % (uncompounded):** 0.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 4 | 33.3% | 2 | 8 | 2 | 0.24% | 2.8% |
| BUY @ 2nd Alert (retest1) | 12 | 4 | 33.3% | 2 | 8 | 2 | 0.24% | 2.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 1 | 11.1% | 0 | 8 | 1 | -0.27% | -2.4% |
| SELL @ 2nd Alert (retest1) | 9 | 1 | 11.1% | 0 | 8 | 1 | -0.27% | -2.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 21 | 5 | 23.8% | 2 | 16 | 3 | 0.02% | 0.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 11:15:00 | 977.15 | 981.35 | 0.00 | ORB-short ORB[978.05,988.70] vol=1.7x ATR=3.48 |
| Stop hit — per-position SL triggered | 2024-05-16 11:20:00 | 980.63 | 981.29 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-06-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 09:30:00 | 1347.45 | 1338.47 | 0.00 | ORB-long ORB[1328.40,1343.00] vol=1.8x ATR=5.92 |
| Stop hit — per-position SL triggered | 2024-06-12 09:35:00 | 1341.53 | 1340.61 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 11:15:00 | 1362.60 | 1373.15 | 0.00 | ORB-short ORB[1368.00,1384.80] vol=2.1x ATR=4.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 11:20:00 | 1355.83 | 1372.32 | 0.00 | T1 1.5R @ 1355.83 |
| Stop hit — per-position SL triggered | 2024-06-13 11:30:00 | 1362.60 | 1371.75 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-08-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 11:05:00 | 1750.40 | 1768.50 | 0.00 | ORB-short ORB[1769.80,1788.95] vol=2.4x ATR=6.85 |
| Stop hit — per-position SL triggered | 2024-08-29 13:25:00 | 1757.25 | 1761.24 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-09-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-27 09:55:00 | 1699.95 | 1710.99 | 0.00 | ORB-short ORB[1705.35,1723.00] vol=2.1x ATR=6.05 |
| Stop hit — per-position SL triggered | 2024-09-27 10:05:00 | 1706.00 | 1710.58 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-11-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-07 09:35:00 | 1603.60 | 1588.67 | 0.00 | ORB-long ORB[1576.00,1597.25] vol=2.0x ATR=8.15 |
| Stop hit — per-position SL triggered | 2024-11-07 09:55:00 | 1595.45 | 1594.27 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-11-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 09:30:00 | 1435.10 | 1428.75 | 0.00 | ORB-long ORB[1417.00,1434.85] vol=2.7x ATR=7.84 |
| Stop hit — per-position SL triggered | 2024-11-25 09:40:00 | 1427.26 | 1429.01 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-12-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 10:00:00 | 1788.15 | 1755.74 | 0.00 | ORB-long ORB[1723.05,1738.60] vol=9.7x ATR=12.15 |
| Stop hit — per-position SL triggered | 2024-12-12 10:25:00 | 1776.00 | 1769.34 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-12-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 09:35:00 | 1735.00 | 1746.16 | 0.00 | ORB-short ORB[1740.70,1765.00] vol=1.7x ATR=7.50 |
| Stop hit — per-position SL triggered | 2024-12-13 09:40:00 | 1742.50 | 1745.18 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-12-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 09:35:00 | 1727.90 | 1742.08 | 0.00 | ORB-short ORB[1731.45,1757.00] vol=2.1x ATR=9.04 |
| Stop hit — per-position SL triggered | 2024-12-17 10:05:00 | 1736.94 | 1739.09 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-12-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-24 09:30:00 | 1549.30 | 1560.31 | 0.00 | ORB-short ORB[1556.20,1574.95] vol=2.8x ATR=8.28 |
| Stop hit — per-position SL triggered | 2024-12-24 09:40:00 | 1557.58 | 1558.86 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-12-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 11:00:00 | 1538.00 | 1551.33 | 0.00 | ORB-short ORB[1547.60,1564.95] vol=2.0x ATR=4.77 |
| Stop hit — per-position SL triggered | 2024-12-27 11:15:00 | 1542.77 | 1550.93 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-01-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:55:00 | 1633.35 | 1624.80 | 0.00 | ORB-long ORB[1606.00,1629.90] vol=1.9x ATR=6.48 |
| Stop hit — per-position SL triggered | 2025-01-01 11:25:00 | 1626.87 | 1626.40 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-01-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 10:05:00 | 1674.60 | 1656.91 | 0.00 | ORB-long ORB[1637.80,1654.90] vol=5.2x ATR=10.46 |
| Stop hit — per-position SL triggered | 2025-01-02 10:15:00 | 1664.14 | 1658.48 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-01-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 09:55:00 | 1670.00 | 1656.07 | 0.00 | ORB-long ORB[1639.00,1663.15] vol=2.8x ATR=8.42 |
| Stop hit — per-position SL triggered | 2025-01-03 11:15:00 | 1661.58 | 1662.47 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-01-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 09:35:00 | 1447.00 | 1440.57 | 0.00 | ORB-long ORB[1429.10,1444.85] vol=1.6x ATR=8.57 |
| Stop hit — per-position SL triggered | 2025-01-16 10:10:00 | 1438.43 | 1441.96 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-01-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 09:35:00 | 1547.70 | 1535.99 | 0.00 | ORB-long ORB[1522.25,1536.85] vol=2.2x ATR=10.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-31 10:05:00 | 1564.05 | 1545.65 | 0.00 | T1 1.5R @ 1564.05 |
| Target hit | 2025-01-31 15:20:00 | 1625.25 | 1611.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — BUY (started 2025-04-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:40:00 | 1738.70 | 1727.97 | 0.00 | ORB-long ORB[1717.00,1738.10] vol=2.0x ATR=7.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 09:45:00 | 1749.76 | 1735.46 | 0.00 | T1 1.5R @ 1749.76 |
| Target hit | 2025-04-21 10:05:00 | 1745.80 | 1757.35 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-16 11:15:00 | 977.15 | 2024-05-16 11:20:00 | 980.63 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-06-12 09:30:00 | 1347.45 | 2024-06-12 09:35:00 | 1341.53 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-06-13 11:15:00 | 1362.60 | 2024-06-13 11:20:00 | 1355.83 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-06-13 11:15:00 | 1362.60 | 2024-06-13 11:30:00 | 1362.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-29 11:05:00 | 1750.40 | 2024-08-29 13:25:00 | 1757.25 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-09-27 09:55:00 | 1699.95 | 2024-09-27 10:05:00 | 1706.00 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-11-07 09:35:00 | 1603.60 | 2024-11-07 09:55:00 | 1595.45 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2024-11-25 09:30:00 | 1435.10 | 2024-11-25 09:40:00 | 1427.26 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2024-12-12 10:00:00 | 1788.15 | 2024-12-12 10:25:00 | 1776.00 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest1 | 2024-12-13 09:35:00 | 1735.00 | 2024-12-13 09:40:00 | 1742.50 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-12-17 09:35:00 | 1727.90 | 2024-12-17 10:05:00 | 1736.94 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2024-12-24 09:30:00 | 1549.30 | 2024-12-24 09:40:00 | 1557.58 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2024-12-27 11:00:00 | 1538.00 | 2024-12-27 11:15:00 | 1542.77 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-01-01 10:55:00 | 1633.35 | 2025-01-01 11:25:00 | 1626.87 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-01-02 10:05:00 | 1674.60 | 2025-01-02 10:15:00 | 1664.14 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest1 | 2025-01-03 09:55:00 | 1670.00 | 2025-01-03 11:15:00 | 1661.58 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-01-16 09:35:00 | 1447.00 | 2025-01-16 10:10:00 | 1438.43 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest1 | 2025-01-31 09:35:00 | 1547.70 | 2025-01-31 10:05:00 | 1564.05 | PARTIAL | 0.50 | 1.06% |
| BUY | retest1 | 2025-01-31 09:35:00 | 1547.70 | 2025-01-31 15:20:00 | 1625.25 | TARGET_HIT | 0.50 | 5.01% |
| BUY | retest1 | 2025-04-21 09:40:00 | 1738.70 | 2025-04-21 09:45:00 | 1749.76 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-04-21 09:40:00 | 1738.70 | 2025-04-21 10:05:00 | 1745.80 | TARGET_HIT | 0.50 | 0.41% |
