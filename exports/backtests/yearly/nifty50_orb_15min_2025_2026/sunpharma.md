# SUNPHARMA (SUNPHARMA)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 1845.00
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
| ENTRY1 | 69 |
| ENTRY2 | 0 |
| PARTIAL | 24 |
| TARGET_HIT | 10 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 93 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 34 / 59
- **Target hits / Stop hits / Partials:** 10 / 59 / 24
- **Avg / median % per leg:** 0.06% / 0.00%
- **Sum % (uncompounded):** 5.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 51 | 17 | 33.3% | 5 | 34 | 12 | 0.04% | 2.1% |
| BUY @ 2nd Alert (retest1) | 51 | 17 | 33.3% | 5 | 34 | 12 | 0.04% | 2.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 42 | 17 | 40.5% | 5 | 25 | 12 | 0.07% | 3.1% |
| SELL @ 2nd Alert (retest1) | 42 | 17 | 40.5% | 5 | 25 | 12 | 0.07% | 3.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 93 | 34 | 36.6% | 10 | 59 | 24 | 0.06% | 5.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-15 09:45:00 | 1688.60 | 1694.87 | 0.00 | ORB-short ORB[1693.50,1708.00] vol=1.8x ATR=3.92 |
| Stop hit — per-position SL triggered | 2025-05-15 09:50:00 | 1692.52 | 1694.63 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-19 09:55:00 | 1764.40 | 1746.93 | 0.00 | ORB-long ORB[1732.20,1749.00] vol=2.0x ATR=5.79 |
| Stop hit — per-position SL triggered | 2025-05-19 10:00:00 | 1758.61 | 1748.91 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-29 11:00:00 | 1682.30 | 1678.85 | 0.00 | ORB-long ORB[1670.40,1681.90] vol=2.2x ATR=2.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-29 11:25:00 | 1686.49 | 1679.93 | 0.00 | T1 1.5R @ 1686.49 |
| Target hit | 2025-05-29 15:20:00 | 1698.60 | 1690.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2025-06-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 09:35:00 | 1669.60 | 1675.22 | 0.00 | ORB-short ORB[1670.90,1695.00] vol=1.7x ATR=4.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 10:35:00 | 1663.51 | 1671.27 | 0.00 | T1 1.5R @ 1663.51 |
| Stop hit — per-position SL triggered | 2025-06-03 11:30:00 | 1669.60 | 1669.29 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-06 09:35:00 | 1675.10 | 1678.62 | 0.00 | ORB-short ORB[1678.70,1689.00] vol=1.6x ATR=3.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-06 11:20:00 | 1669.38 | 1674.79 | 0.00 | T1 1.5R @ 1669.38 |
| Stop hit — per-position SL triggered | 2025-06-06 11:45:00 | 1675.10 | 1674.31 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-09 10:00:00 | 1677.00 | 1679.47 | 0.00 | ORB-short ORB[1679.90,1689.00] vol=2.3x ATR=2.68 |
| Stop hit — per-position SL triggered | 2025-06-09 10:55:00 | 1679.68 | 1678.39 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-10 11:15:00 | 1683.30 | 1691.30 | 0.00 | ORB-short ORB[1690.20,1706.90] vol=1.7x ATR=2.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-10 12:15:00 | 1679.96 | 1688.98 | 0.00 | T1 1.5R @ 1679.96 |
| Stop hit — per-position SL triggered | 2025-06-10 12:45:00 | 1683.30 | 1687.45 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-13 11:00:00 | 1686.70 | 1678.69 | 0.00 | ORB-long ORB[1661.10,1683.00] vol=1.8x ATR=3.58 |
| Stop hit — per-position SL triggered | 2025-06-13 11:10:00 | 1683.12 | 1678.92 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-17 09:50:00 | 1654.60 | 1664.42 | 0.00 | ORB-short ORB[1664.40,1681.70] vol=1.8x ATR=3.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-17 10:15:00 | 1649.01 | 1659.69 | 0.00 | T1 1.5R @ 1649.01 |
| Stop hit — per-position SL triggered | 2025-06-17 10:20:00 | 1654.60 | 1659.42 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 10:40:00 | 1660.10 | 1670.52 | 0.00 | ORB-short ORB[1666.30,1678.90] vol=2.6x ATR=3.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 11:15:00 | 1654.89 | 1667.06 | 0.00 | T1 1.5R @ 1654.89 |
| Stop hit — per-position SL triggered | 2025-06-26 12:30:00 | 1660.10 | 1662.44 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 10:55:00 | 1701.20 | 1693.78 | 0.00 | ORB-long ORB[1685.70,1698.10] vol=1.7x ATR=3.30 |
| Stop hit — per-position SL triggered | 2025-06-30 12:10:00 | 1697.90 | 1696.63 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:15:00 | 1659.10 | 1664.28 | 0.00 | ORB-short ORB[1666.70,1684.30] vol=1.9x ATR=3.29 |
| Stop hit — per-position SL triggered | 2025-07-01 10:25:00 | 1662.39 | 1664.17 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 11:05:00 | 1659.10 | 1666.24 | 0.00 | ORB-short ORB[1667.10,1682.00] vol=2.3x ATR=2.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 11:20:00 | 1654.86 | 1664.46 | 0.00 | T1 1.5R @ 1654.86 |
| Stop hit — per-position SL triggered | 2025-07-08 12:10:00 | 1659.10 | 1662.19 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:40:00 | 1690.40 | 1694.62 | 0.00 | ORB-short ORB[1692.40,1709.40] vol=1.7x ATR=2.76 |
| Stop hit — per-position SL triggered | 2025-07-18 11:00:00 | 1693.16 | 1694.15 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 09:50:00 | 1691.70 | 1683.48 | 0.00 | ORB-long ORB[1677.50,1689.40] vol=2.3x ATR=4.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 10:05:00 | 1698.13 | 1684.88 | 0.00 | T1 1.5R @ 1698.13 |
| Stop hit — per-position SL triggered | 2025-07-21 10:35:00 | 1691.70 | 1686.82 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 10:25:00 | 1730.00 | 1725.70 | 0.00 | ORB-long ORB[1712.10,1724.80] vol=5.3x ATR=3.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-30 10:50:00 | 1734.68 | 1728.19 | 0.00 | T1 1.5R @ 1734.68 |
| Stop hit — per-position SL triggered | 2025-07-30 11:20:00 | 1730.00 | 1728.66 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-08-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-11 10:30:00 | 1598.90 | 1592.55 | 0.00 | ORB-long ORB[1584.20,1594.60] vol=1.7x ATR=3.17 |
| Stop hit — per-position SL triggered | 2025-08-11 10:35:00 | 1595.73 | 1593.10 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-18 11:15:00 | 1628.60 | 1636.85 | 0.00 | ORB-short ORB[1632.20,1649.60] vol=1.6x ATR=2.43 |
| Stop hit — per-position SL triggered | 2025-08-18 12:00:00 | 1631.03 | 1634.06 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-08-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 10:50:00 | 1648.30 | 1635.86 | 0.00 | ORB-long ORB[1623.90,1636.00] vol=3.6x ATR=3.48 |
| Stop hit — per-position SL triggered | 2025-08-21 11:05:00 | 1644.82 | 1636.62 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-08-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 09:50:00 | 1650.90 | 1641.70 | 0.00 | ORB-long ORB[1635.60,1642.00] vol=2.2x ATR=3.53 |
| Stop hit — per-position SL triggered | 2025-08-22 10:05:00 | 1647.37 | 1644.15 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-09-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-01 10:35:00 | 1584.10 | 1588.41 | 0.00 | ORB-short ORB[1586.50,1600.20] vol=2.0x ATR=3.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 11:45:00 | 1579.40 | 1585.03 | 0.00 | T1 1.5R @ 1579.40 |
| Target hit | 2025-09-01 15:20:00 | 1563.30 | 1571.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 11:15:00 | 1577.50 | 1567.13 | 0.00 | ORB-long ORB[1558.30,1569.30] vol=1.8x ATR=2.85 |
| Stop hit — per-position SL triggered | 2025-09-03 11:20:00 | 1574.65 | 1567.47 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-09-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-05 10:25:00 | 1588.10 | 1586.05 | 0.00 | ORB-long ORB[1575.50,1582.90] vol=5.2x ATR=3.34 |
| Stop hit — per-position SL triggered | 2025-09-05 10:30:00 | 1584.76 | 1586.06 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-09-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 10:30:00 | 1603.60 | 1598.92 | 0.00 | ORB-long ORB[1585.30,1598.80] vol=1.7x ATR=3.32 |
| Stop hit — per-position SL triggered | 2025-09-08 10:35:00 | 1600.28 | 1599.15 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-09-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 10:20:00 | 1598.40 | 1593.11 | 0.00 | ORB-long ORB[1582.40,1593.90] vol=3.3x ATR=3.51 |
| Stop hit — per-position SL triggered | 2025-09-10 10:40:00 | 1594.89 | 1593.23 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 11:05:00 | 1609.20 | 1604.78 | 0.00 | ORB-long ORB[1600.00,1606.90] vol=1.8x ATR=1.98 |
| Stop hit — per-position SL triggered | 2025-09-16 11:20:00 | 1607.22 | 1605.04 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-09-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 11:10:00 | 1654.50 | 1650.32 | 0.00 | ORB-long ORB[1643.10,1652.00] vol=3.3x ATR=2.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 11:35:00 | 1658.39 | 1651.99 | 0.00 | T1 1.5R @ 1658.39 |
| Stop hit — per-position SL triggered | 2025-09-19 15:00:00 | 1654.50 | 1657.28 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-09-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-25 10:55:00 | 1639.80 | 1632.08 | 0.00 | ORB-long ORB[1624.00,1638.30] vol=2.5x ATR=3.06 |
| Stop hit — per-position SL triggered | 2025-09-25 12:40:00 | 1636.74 | 1636.96 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-09-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 09:45:00 | 1617.40 | 1600.54 | 0.00 | ORB-long ORB[1580.00,1599.90] vol=2.6x ATR=5.84 |
| Stop hit — per-position SL triggered | 2025-09-29 10:05:00 | 1611.56 | 1603.18 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-10-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 11:10:00 | 1637.90 | 1634.64 | 0.00 | ORB-long ORB[1620.00,1637.30] vol=2.1x ATR=2.69 |
| Stop hit — per-position SL triggered | 2025-10-06 11:20:00 | 1635.21 | 1634.68 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-10-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 11:10:00 | 1658.30 | 1654.10 | 0.00 | ORB-long ORB[1647.00,1658.00] vol=4.9x ATR=2.81 |
| Stop hit — per-position SL triggered | 2025-10-07 11:15:00 | 1655.49 | 1654.25 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-10-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 11:10:00 | 1663.90 | 1659.99 | 0.00 | ORB-long ORB[1649.00,1657.90] vol=1.9x ATR=3.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-10 11:55:00 | 1668.59 | 1661.66 | 0.00 | T1 1.5R @ 1668.59 |
| Target hit | 2025-10-10 15:20:00 | 1670.30 | 1667.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 11:15:00 | 1655.90 | 1661.15 | 0.00 | ORB-short ORB[1657.10,1669.90] vol=2.0x ATR=2.67 |
| Stop hit — per-position SL triggered | 2025-10-14 11:20:00 | 1658.57 | 1660.36 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-10-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 11:00:00 | 1674.50 | 1668.71 | 0.00 | ORB-long ORB[1656.70,1664.50] vol=2.9x ATR=2.72 |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 1671.78 | 1669.56 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-10-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 09:30:00 | 1700.60 | 1694.62 | 0.00 | ORB-long ORB[1685.70,1698.90] vol=1.6x ATR=3.65 |
| Stop hit — per-position SL triggered | 2025-10-28 10:00:00 | 1696.95 | 1696.78 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-10-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:55:00 | 1713.60 | 1699.71 | 0.00 | ORB-long ORB[1690.00,1700.70] vol=3.0x ATR=3.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 13:20:00 | 1719.04 | 1708.03 | 0.00 | T1 1.5R @ 1719.04 |
| Stop hit — per-position SL triggered | 2025-10-29 14:00:00 | 1713.60 | 1709.58 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-11-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 10:50:00 | 1698.10 | 1694.35 | 0.00 | ORB-long ORB[1681.20,1694.50] vol=1.7x ATR=3.09 |
| Stop hit — per-position SL triggered | 2025-11-03 11:20:00 | 1695.01 | 1694.60 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-11-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 10:10:00 | 1732.50 | 1720.88 | 0.00 | ORB-long ORB[1712.00,1723.80] vol=3.6x ATR=4.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 12:15:00 | 1739.33 | 1729.98 | 0.00 | T1 1.5R @ 1739.33 |
| Stop hit — per-position SL triggered | 2025-11-12 12:45:00 | 1732.50 | 1731.99 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-11-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 10:50:00 | 1746.30 | 1744.46 | 0.00 | ORB-long ORB[1734.70,1745.00] vol=4.1x ATR=3.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-14 11:20:00 | 1751.65 | 1745.95 | 0.00 | T1 1.5R @ 1751.65 |
| Stop hit — per-position SL triggered | 2025-11-14 14:55:00 | 1746.30 | 1749.59 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-11-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-19 10:35:00 | 1764.60 | 1757.20 | 0.00 | ORB-long ORB[1751.10,1763.00] vol=1.7x ATR=2.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 10:40:00 | 1768.86 | 1760.07 | 0.00 | T1 1.5R @ 1768.86 |
| Target hit | 2025-11-19 15:20:00 | 1784.90 | 1780.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — BUY (started 2025-11-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 11:05:00 | 1788.10 | 1785.09 | 0.00 | ORB-long ORB[1773.60,1785.00] vol=6.9x ATR=3.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-26 11:40:00 | 1793.31 | 1786.30 | 0.00 | T1 1.5R @ 1793.31 |
| Target hit | 2025-11-26 15:20:00 | 1805.90 | 1796.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — BUY (started 2025-11-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 09:50:00 | 1820.00 | 1815.48 | 0.00 | ORB-long ORB[1807.50,1814.40] vol=2.4x ATR=2.99 |
| Stop hit — per-position SL triggered | 2025-11-28 10:50:00 | 1817.01 | 1817.57 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-12-01 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 10:20:00 | 1819.70 | 1829.36 | 0.00 | ORB-short ORB[1831.60,1849.00] vol=1.9x ATR=3.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 11:45:00 | 1814.28 | 1823.61 | 0.00 | T1 1.5R @ 1814.28 |
| Target hit | 2025-12-01 15:20:00 | 1810.00 | 1813.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — SELL (started 2025-12-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 11:10:00 | 1785.30 | 1793.00 | 0.00 | ORB-short ORB[1796.00,1816.00] vol=1.9x ATR=3.40 |
| Stop hit — per-position SL triggered | 2025-12-03 11:25:00 | 1788.70 | 1791.95 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-12-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 10:50:00 | 1806.00 | 1809.68 | 0.00 | ORB-short ORB[1810.40,1820.20] vol=1.8x ATR=2.89 |
| Stop hit — per-position SL triggered | 2025-12-05 11:00:00 | 1808.89 | 1809.49 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-12-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 10:40:00 | 1786.90 | 1787.67 | 0.00 | ORB-short ORB[1789.30,1801.20] vol=3.9x ATR=3.87 |
| Stop hit — per-position SL triggered | 2025-12-09 11:10:00 | 1790.77 | 1787.71 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-12-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 10:45:00 | 1795.40 | 1785.97 | 0.00 | ORB-long ORB[1775.10,1788.60] vol=3.4x ATR=3.79 |
| Stop hit — per-position SL triggered | 2025-12-11 10:55:00 | 1791.61 | 1787.40 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-12-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-12 11:00:00 | 1794.50 | 1801.47 | 0.00 | ORB-short ORB[1801.20,1811.80] vol=2.8x ATR=2.56 |
| Stop hit — per-position SL triggered | 2025-12-12 11:10:00 | 1797.06 | 1799.25 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-17 11:15:00 | 1775.20 | 1777.36 | 0.00 | ORB-short ORB[1776.60,1785.50] vol=2.5x ATR=2.08 |
| Stop hit — per-position SL triggered | 2025-12-17 11:25:00 | 1777.28 | 1777.17 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-12-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 10:20:00 | 1739.90 | 1745.44 | 0.00 | ORB-short ORB[1747.80,1759.00] vol=2.2x ATR=3.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 10:30:00 | 1735.08 | 1744.08 | 0.00 | T1 1.5R @ 1735.08 |
| Target hit | 2025-12-24 12:40:00 | 1735.80 | 1735.27 | 0.00 | Trail-exit close>VWAP |

### Cycle 51 — SELL (started 2025-12-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 09:35:00 | 1716.20 | 1723.07 | 0.00 | ORB-short ORB[1723.40,1737.00] vol=1.7x ATR=3.42 |
| Stop hit — per-position SL triggered | 2025-12-26 09:55:00 | 1719.62 | 1720.01 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2026-01-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-01 10:55:00 | 1711.80 | 1714.18 | 0.00 | ORB-short ORB[1714.10,1724.70] vol=1.6x ATR=2.50 |
| Stop hit — per-position SL triggered | 2026-01-01 11:25:00 | 1714.30 | 1713.16 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2026-01-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 11:00:00 | 1731.40 | 1725.31 | 0.00 | ORB-long ORB[1719.40,1729.60] vol=1.8x ATR=1.86 |
| Stop hit — per-position SL triggered | 2026-01-02 11:40:00 | 1729.54 | 1726.24 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2026-01-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 10:55:00 | 1744.90 | 1737.01 | 0.00 | ORB-long ORB[1723.40,1732.90] vol=1.8x ATR=2.77 |
| Stop hit — per-position SL triggered | 2026-01-05 11:25:00 | 1742.13 | 1738.15 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2026-01-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 10:50:00 | 1743.80 | 1738.44 | 0.00 | ORB-long ORB[1725.70,1741.80] vol=3.2x ATR=3.03 |
| Stop hit — per-position SL triggered | 2026-01-06 11:00:00 | 1740.77 | 1738.72 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2026-01-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 09:55:00 | 1775.60 | 1760.81 | 0.00 | ORB-long ORB[1748.50,1757.20] vol=1.7x ATR=4.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-07 10:00:00 | 1782.55 | 1768.51 | 0.00 | T1 1.5R @ 1782.55 |
| Target hit | 2026-01-07 10:40:00 | 1781.40 | 1785.13 | 0.00 | Trail-exit close<VWAP |

### Cycle 57 — SELL (started 2026-01-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-12 10:35:00 | 1719.50 | 1724.03 | 0.00 | ORB-short ORB[1720.80,1739.30] vol=2.1x ATR=4.53 |
| Stop hit — per-position SL triggered | 2026-01-12 10:45:00 | 1724.03 | 1723.93 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2026-01-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-21 11:00:00 | 1620.30 | 1628.37 | 0.00 | ORB-short ORB[1621.60,1645.00] vol=2.5x ATR=4.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 13:10:00 | 1613.68 | 1623.17 | 0.00 | T1 1.5R @ 1613.68 |
| Target hit | 2026-01-21 15:20:00 | 1613.60 | 1620.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — SELL (started 2026-02-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:10:00 | 1726.30 | 1727.22 | 0.00 | ORB-short ORB[1726.50,1733.00] vol=1.8x ATR=2.33 |
| Stop hit — per-position SL triggered | 2026-02-24 10:30:00 | 1728.63 | 1727.15 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2026-03-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:25:00 | 1816.60 | 1821.63 | 0.00 | ORB-short ORB[1820.90,1830.90] vol=1.9x ATR=4.22 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 1820.82 | 1821.34 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2026-03-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 11:15:00 | 1782.60 | 1794.21 | 0.00 | ORB-short ORB[1798.30,1818.00] vol=2.7x ATR=4.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 11:35:00 | 1775.67 | 1792.64 | 0.00 | T1 1.5R @ 1775.67 |
| Stop hit — per-position SL triggered | 2026-03-16 14:25:00 | 1782.60 | 1783.99 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2026-04-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 11:00:00 | 1758.70 | 1774.63 | 0.00 | ORB-short ORB[1762.00,1787.20] vol=2.9x ATR=4.99 |
| Stop hit — per-position SL triggered | 2026-04-01 11:05:00 | 1763.69 | 1774.37 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2026-04-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:45:00 | 1675.80 | 1669.18 | 0.00 | ORB-long ORB[1657.70,1675.00] vol=1.7x ATR=2.93 |
| Stop hit — per-position SL triggered | 2026-04-15 11:00:00 | 1672.87 | 1669.58 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2026-04-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 11:05:00 | 1679.30 | 1688.13 | 0.00 | ORB-short ORB[1682.10,1696.90] vol=1.6x ATR=3.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 12:30:00 | 1674.24 | 1684.33 | 0.00 | T1 1.5R @ 1674.24 |
| Target hit | 2026-04-17 15:20:00 | 1676.90 | 1678.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 65 — SELL (started 2026-04-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:55:00 | 1654.60 | 1660.02 | 0.00 | ORB-short ORB[1657.20,1672.00] vol=2.0x ATR=2.84 |
| Stop hit — per-position SL triggered | 2026-04-22 11:00:00 | 1657.44 | 1659.94 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2026-04-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:40:00 | 1693.80 | 1680.44 | 0.00 | ORB-long ORB[1661.20,1676.10] vol=1.7x ATR=4.56 |
| Stop hit — per-position SL triggered | 2026-04-23 09:55:00 | 1689.24 | 1684.91 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-04-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:40:00 | 1736.00 | 1729.68 | 0.00 | ORB-long ORB[1714.30,1735.50] vol=3.4x ATR=4.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 09:45:00 | 1742.72 | 1731.34 | 0.00 | T1 1.5R @ 1742.72 |
| Stop hit — per-position SL triggered | 2026-04-28 09:55:00 | 1736.00 | 1731.75 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2026-05-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:55:00 | 1844.30 | 1830.19 | 0.00 | ORB-long ORB[1818.40,1837.00] vol=3.3x ATR=5.05 |
| Stop hit — per-position SL triggered | 2026-05-06 11:05:00 | 1839.25 | 1833.04 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2026-05-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 11:05:00 | 1849.70 | 1841.48 | 0.00 | ORB-long ORB[1819.20,1844.10] vol=2.6x ATR=3.91 |
| Stop hit — per-position SL triggered | 2026-05-08 12:00:00 | 1845.79 | 1843.24 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-15 09:45:00 | 1688.60 | 2025-05-15 09:50:00 | 1692.52 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-05-19 09:55:00 | 1764.40 | 2025-05-19 10:00:00 | 1758.61 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-05-29 11:00:00 | 1682.30 | 2025-05-29 11:25:00 | 1686.49 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2025-05-29 11:00:00 | 1682.30 | 2025-05-29 15:20:00 | 1698.60 | TARGET_HIT | 0.50 | 0.97% |
| SELL | retest1 | 2025-06-03 09:35:00 | 1669.60 | 2025-06-03 10:35:00 | 1663.51 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-06-03 09:35:00 | 1669.60 | 2025-06-03 11:30:00 | 1669.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-06 09:35:00 | 1675.10 | 2025-06-06 11:20:00 | 1669.38 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-06-06 09:35:00 | 1675.10 | 2025-06-06 11:45:00 | 1675.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-09 10:00:00 | 1677.00 | 2025-06-09 10:55:00 | 1679.68 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-06-10 11:15:00 | 1683.30 | 2025-06-10 12:15:00 | 1679.96 | PARTIAL | 0.50 | 0.20% |
| SELL | retest1 | 2025-06-10 11:15:00 | 1683.30 | 2025-06-10 12:45:00 | 1683.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-13 11:00:00 | 1686.70 | 2025-06-13 11:10:00 | 1683.12 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-06-17 09:50:00 | 1654.60 | 2025-06-17 10:15:00 | 1649.01 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-06-17 09:50:00 | 1654.60 | 2025-06-17 10:20:00 | 1654.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-26 10:40:00 | 1660.10 | 2025-06-26 11:15:00 | 1654.89 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-06-26 10:40:00 | 1660.10 | 2025-06-26 12:30:00 | 1660.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-30 10:55:00 | 1701.20 | 2025-06-30 12:10:00 | 1697.90 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-07-01 10:15:00 | 1659.10 | 2025-07-01 10:25:00 | 1662.39 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-07-08 11:05:00 | 1659.10 | 2025-07-08 11:20:00 | 1654.86 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-07-08 11:05:00 | 1659.10 | 2025-07-08 12:10:00 | 1659.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-18 10:40:00 | 1690.40 | 2025-07-18 11:00:00 | 1693.16 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-07-21 09:50:00 | 1691.70 | 2025-07-21 10:05:00 | 1698.13 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-07-21 09:50:00 | 1691.70 | 2025-07-21 10:35:00 | 1691.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-30 10:25:00 | 1730.00 | 2025-07-30 10:50:00 | 1734.68 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2025-07-30 10:25:00 | 1730.00 | 2025-07-30 11:20:00 | 1730.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-11 10:30:00 | 1598.90 | 2025-08-11 10:35:00 | 1595.73 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-08-18 11:15:00 | 1628.60 | 2025-08-18 12:00:00 | 1631.03 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-08-21 10:50:00 | 1648.30 | 2025-08-21 11:05:00 | 1644.82 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-08-22 09:50:00 | 1650.90 | 2025-08-22 10:05:00 | 1647.37 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-09-01 10:35:00 | 1584.10 | 2025-09-01 11:45:00 | 1579.40 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-09-01 10:35:00 | 1584.10 | 2025-09-01 15:20:00 | 1563.30 | TARGET_HIT | 0.50 | 1.31% |
| BUY | retest1 | 2025-09-03 11:15:00 | 1577.50 | 2025-09-03 11:20:00 | 1574.65 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-09-05 10:25:00 | 1588.10 | 2025-09-05 10:30:00 | 1584.76 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-09-08 10:30:00 | 1603.60 | 2025-09-08 10:35:00 | 1600.28 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-09-10 10:20:00 | 1598.40 | 2025-09-10 10:40:00 | 1594.89 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-09-16 11:05:00 | 1609.20 | 2025-09-16 11:20:00 | 1607.22 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest1 | 2025-09-19 11:10:00 | 1654.50 | 2025-09-19 11:35:00 | 1658.39 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2025-09-19 11:10:00 | 1654.50 | 2025-09-19 15:00:00 | 1654.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-25 10:55:00 | 1639.80 | 2025-09-25 12:40:00 | 1636.74 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-09-29 09:45:00 | 1617.40 | 2025-09-29 10:05:00 | 1611.56 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-10-06 11:10:00 | 1637.90 | 2025-10-06 11:20:00 | 1635.21 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-10-07 11:10:00 | 1658.30 | 2025-10-07 11:15:00 | 1655.49 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-10-10 11:10:00 | 1663.90 | 2025-10-10 11:55:00 | 1668.59 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-10-10 11:10:00 | 1663.90 | 2025-10-10 15:20:00 | 1670.30 | TARGET_HIT | 0.50 | 0.38% |
| SELL | retest1 | 2025-10-14 11:15:00 | 1655.90 | 2025-10-14 11:20:00 | 1658.57 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-10-17 11:00:00 | 1674.50 | 2025-10-17 11:15:00 | 1671.78 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-10-28 09:30:00 | 1700.60 | 2025-10-28 10:00:00 | 1696.95 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-10-29 10:55:00 | 1713.60 | 2025-10-29 13:20:00 | 1719.04 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-10-29 10:55:00 | 1713.60 | 2025-10-29 14:00:00 | 1713.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-03 10:50:00 | 1698.10 | 2025-11-03 11:20:00 | 1695.01 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-11-12 10:10:00 | 1732.50 | 2025-11-12 12:15:00 | 1739.33 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-11-12 10:10:00 | 1732.50 | 2025-11-12 12:45:00 | 1732.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-14 10:50:00 | 1746.30 | 2025-11-14 11:20:00 | 1751.65 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-11-14 10:50:00 | 1746.30 | 2025-11-14 14:55:00 | 1746.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-19 10:35:00 | 1764.60 | 2025-11-19 10:40:00 | 1768.86 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2025-11-19 10:35:00 | 1764.60 | 2025-11-19 15:20:00 | 1784.90 | TARGET_HIT | 0.50 | 1.15% |
| BUY | retest1 | 2025-11-26 11:05:00 | 1788.10 | 2025-11-26 11:40:00 | 1793.31 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-11-26 11:05:00 | 1788.10 | 2025-11-26 15:20:00 | 1805.90 | TARGET_HIT | 0.50 | 1.00% |
| BUY | retest1 | 2025-11-28 09:50:00 | 1820.00 | 2025-11-28 10:50:00 | 1817.01 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-12-01 10:20:00 | 1819.70 | 2025-12-01 11:45:00 | 1814.28 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-12-01 10:20:00 | 1819.70 | 2025-12-01 15:20:00 | 1810.00 | TARGET_HIT | 0.50 | 0.53% |
| SELL | retest1 | 2025-12-03 11:10:00 | 1785.30 | 2025-12-03 11:25:00 | 1788.70 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-12-05 10:50:00 | 1806.00 | 2025-12-05 11:00:00 | 1808.89 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-12-09 10:40:00 | 1786.90 | 2025-12-09 11:10:00 | 1790.77 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-12-11 10:45:00 | 1795.40 | 2025-12-11 10:55:00 | 1791.61 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-12-12 11:00:00 | 1794.50 | 2025-12-12 11:10:00 | 1797.06 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-12-17 11:15:00 | 1775.20 | 2025-12-17 11:25:00 | 1777.28 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest1 | 2025-12-24 10:20:00 | 1739.90 | 2025-12-24 10:30:00 | 1735.08 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-12-24 10:20:00 | 1739.90 | 2025-12-24 12:40:00 | 1735.80 | TARGET_HIT | 0.50 | 0.24% |
| SELL | retest1 | 2025-12-26 09:35:00 | 1716.20 | 2025-12-26 09:55:00 | 1719.62 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-01-01 10:55:00 | 1711.80 | 2026-01-01 11:25:00 | 1714.30 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2026-01-02 11:00:00 | 1731.40 | 2026-01-02 11:40:00 | 1729.54 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest1 | 2026-01-05 10:55:00 | 1744.90 | 2026-01-05 11:25:00 | 1742.13 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2026-01-06 10:50:00 | 1743.80 | 2026-01-06 11:00:00 | 1740.77 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2026-01-07 09:55:00 | 1775.60 | 2026-01-07 10:00:00 | 1782.55 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-01-07 09:55:00 | 1775.60 | 2026-01-07 10:40:00 | 1781.40 | TARGET_HIT | 0.50 | 0.33% |
| SELL | retest1 | 2026-01-12 10:35:00 | 1719.50 | 2026-01-12 10:45:00 | 1724.03 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-01-21 11:00:00 | 1620.30 | 2026-01-21 13:10:00 | 1613.68 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-01-21 11:00:00 | 1620.30 | 2026-01-21 15:20:00 | 1613.60 | TARGET_HIT | 0.50 | 0.41% |
| SELL | retest1 | 2026-02-24 10:10:00 | 1726.30 | 2026-02-24 10:30:00 | 1728.63 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2026-03-13 10:25:00 | 1816.60 | 2026-03-13 10:50:00 | 1820.82 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-03-16 11:15:00 | 1782.60 | 2026-03-16 11:35:00 | 1775.67 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-03-16 11:15:00 | 1782.60 | 2026-03-16 14:25:00 | 1782.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-01 11:00:00 | 1758.70 | 2026-04-01 11:05:00 | 1763.69 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-15 10:45:00 | 1675.80 | 2026-04-15 11:00:00 | 1672.87 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-04-17 11:05:00 | 1679.30 | 2026-04-17 12:30:00 | 1674.24 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2026-04-17 11:05:00 | 1679.30 | 2026-04-17 15:20:00 | 1676.90 | TARGET_HIT | 0.50 | 0.14% |
| SELL | retest1 | 2026-04-22 10:55:00 | 1654.60 | 2026-04-22 11:00:00 | 1657.44 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2026-04-23 09:40:00 | 1693.80 | 2026-04-23 09:55:00 | 1689.24 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-04-28 09:40:00 | 1736.00 | 2026-04-28 09:45:00 | 1742.72 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-04-28 09:40:00 | 1736.00 | 2026-04-28 09:55:00 | 1736.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-06 10:55:00 | 1844.30 | 2026-05-06 11:05:00 | 1839.25 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-05-08 11:05:00 | 1849.70 | 2026-05-08 12:00:00 | 1845.79 | STOP_HIT | 1.00 | -0.21% |
