# Prestige Estates Projects Ltd. (PRESTIGE)

## Backtest Summary

- **Window:** 2025-08-11 09:15:00 → 2026-05-08 15:25:00 (13588 bars)
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
| ENTRY1 | 50 |
| ENTRY2 | 0 |
| PARTIAL | 17 |
| TARGET_HIT | 8 |
| STOP_HIT | 42 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 67 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 42
- **Target hits / Stop hits / Partials:** 8 / 42 / 17
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 7.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 16 | 40.0% | 6 | 24 | 10 | 0.12% | 4.7% |
| BUY @ 2nd Alert (retest1) | 40 | 16 | 40.0% | 6 | 24 | 10 | 0.12% | 4.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 27 | 9 | 33.3% | 2 | 18 | 7 | 0.11% | 2.9% |
| SELL @ 2nd Alert (retest1) | 27 | 9 | 33.3% | 2 | 18 | 7 | 0.11% | 2.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 67 | 25 | 37.3% | 8 | 42 | 17 | 0.11% | 7.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-08-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-11 10:50:00 | 1615.00 | 1612.01 | 0.00 | ORB-long ORB[1598.00,1614.80] vol=3.7x ATR=7.45 |
| Stop hit — per-position SL triggered | 2025-08-11 11:55:00 | 1607.55 | 1613.86 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-08-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 10:40:00 | 1613.40 | 1615.78 | 0.00 | ORB-short ORB[1619.30,1639.90] vol=1.8x ATR=4.38 |
| Stop hit — per-position SL triggered | 2025-08-14 10:55:00 | 1617.78 | 1615.72 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-08-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 09:40:00 | 1633.70 | 1625.49 | 0.00 | ORB-long ORB[1610.40,1632.00] vol=2.2x ATR=5.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 10:00:00 | 1642.51 | 1630.51 | 0.00 | T1 1.5R @ 1642.51 |
| Target hit | 2025-08-18 10:30:00 | 1633.80 | 1635.26 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — SELL (started 2025-08-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 09:30:00 | 1608.70 | 1618.43 | 0.00 | ORB-short ORB[1612.20,1636.00] vol=2.4x ATR=5.21 |
| Stop hit — per-position SL triggered | 2025-08-19 09:50:00 | 1613.91 | 1613.44 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 10:15:00 | 1658.20 | 1649.90 | 0.00 | ORB-long ORB[1642.00,1657.00] vol=2.1x ATR=4.72 |
| Stop hit — per-position SL triggered | 2025-08-25 12:10:00 | 1653.48 | 1655.89 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-09-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 09:50:00 | 1574.50 | 1569.05 | 0.00 | ORB-long ORB[1553.00,1573.90] vol=11.3x ATR=5.43 |
| Stop hit — per-position SL triggered | 2025-09-03 09:55:00 | 1569.07 | 1569.40 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-09-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:20:00 | 1535.70 | 1555.31 | 0.00 | ORB-short ORB[1556.10,1570.00] vol=1.8x ATR=6.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 10:30:00 | 1525.70 | 1549.25 | 0.00 | T1 1.5R @ 1525.70 |
| Stop hit — per-position SL triggered | 2025-09-05 10:35:00 | 1535.70 | 1548.25 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-09-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 09:50:00 | 1572.10 | 1564.77 | 0.00 | ORB-long ORB[1553.50,1564.40] vol=3.6x ATR=5.17 |
| Stop hit — per-position SL triggered | 2025-09-11 10:45:00 | 1566.93 | 1566.54 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-09-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 09:45:00 | 1611.00 | 1603.06 | 0.00 | ORB-long ORB[1592.40,1604.10] vol=1.8x ATR=4.93 |
| Stop hit — per-position SL triggered | 2025-09-16 10:35:00 | 1606.07 | 1606.45 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 10:15:00 | 1658.40 | 1642.81 | 0.00 | ORB-long ORB[1632.20,1654.00] vol=2.6x ATR=7.09 |
| Stop hit — per-position SL triggered | 2025-09-17 10:30:00 | 1651.31 | 1645.26 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-09-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-22 10:50:00 | 1638.80 | 1623.90 | 0.00 | ORB-long ORB[1604.00,1621.50] vol=2.1x ATR=5.04 |
| Stop hit — per-position SL triggered | 2025-09-22 11:05:00 | 1633.76 | 1624.73 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-09-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 10:20:00 | 1528.50 | 1511.46 | 0.00 | ORB-long ORB[1500.80,1515.50] vol=2.4x ATR=5.61 |
| Stop hit — per-position SL triggered | 2025-09-29 10:30:00 | 1522.89 | 1513.08 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-10-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 11:10:00 | 1530.70 | 1539.78 | 0.00 | ORB-short ORB[1530.90,1551.50] vol=1.8x ATR=3.73 |
| Stop hit — per-position SL triggered | 2025-10-07 11:50:00 | 1534.43 | 1538.17 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-10-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:30:00 | 1606.20 | 1593.72 | 0.00 | ORB-long ORB[1572.00,1590.00] vol=2.1x ATR=7.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-10 09:35:00 | 1616.91 | 1602.63 | 0.00 | T1 1.5R @ 1616.91 |
| Target hit | 2025-10-10 10:20:00 | 1608.20 | 1609.48 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — BUY (started 2025-10-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 10:30:00 | 1629.50 | 1622.78 | 0.00 | ORB-long ORB[1607.00,1624.40] vol=2.3x ATR=6.07 |
| Stop hit — per-position SL triggered | 2025-10-13 12:30:00 | 1623.43 | 1625.70 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-10-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 09:35:00 | 1715.00 | 1703.32 | 0.00 | ORB-long ORB[1680.20,1703.20] vol=4.5x ATR=6.29 |
| Stop hit — per-position SL triggered | 2025-10-17 09:40:00 | 1708.71 | 1707.63 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-10-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 10:50:00 | 1723.10 | 1708.74 | 0.00 | ORB-long ORB[1701.10,1717.80] vol=3.2x ATR=5.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 10:55:00 | 1730.87 | 1716.21 | 0.00 | T1 1.5R @ 1730.87 |
| Stop hit — per-position SL triggered | 2025-10-20 12:10:00 | 1723.10 | 1724.88 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-10-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 09:30:00 | 1732.10 | 1721.14 | 0.00 | ORB-long ORB[1707.70,1725.80] vol=1.7x ATR=5.54 |
| Stop hit — per-position SL triggered | 2025-10-23 09:35:00 | 1726.56 | 1721.57 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-10-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 09:35:00 | 1762.80 | 1750.76 | 0.00 | ORB-long ORB[1728.50,1748.00] vol=1.8x ATR=5.66 |
| Stop hit — per-position SL triggered | 2025-10-24 09:50:00 | 1757.14 | 1754.11 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-10-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 10:00:00 | 1790.30 | 1786.74 | 0.00 | ORB-long ORB[1754.10,1781.00] vol=5.0x ATR=5.82 |
| Stop hit — per-position SL triggered | 2025-10-27 10:10:00 | 1784.48 | 1786.63 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-10-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-29 10:45:00 | 1749.10 | 1751.25 | 0.00 | ORB-short ORB[1750.30,1762.70] vol=1.6x ATR=4.10 |
| Stop hit — per-position SL triggered | 2025-10-29 11:00:00 | 1753.20 | 1751.08 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-10-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 09:55:00 | 1746.20 | 1757.06 | 0.00 | ORB-short ORB[1754.10,1770.30] vol=2.4x ATR=5.56 |
| Stop hit — per-position SL triggered | 2025-10-30 10:00:00 | 1751.76 | 1754.81 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-10-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 09:30:00 | 1771.30 | 1767.22 | 0.00 | ORB-long ORB[1751.20,1769.80] vol=10.8x ATR=4.72 |
| Stop hit — per-position SL triggered | 2025-10-31 09:40:00 | 1766.58 | 1767.46 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-11-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 09:45:00 | 1766.60 | 1760.50 | 0.00 | ORB-long ORB[1735.40,1750.60] vol=7.3x ATR=6.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 11:45:00 | 1777.02 | 1764.35 | 0.00 | T1 1.5R @ 1777.02 |
| Target hit | 2025-11-03 14:10:00 | 1784.40 | 1785.22 | 0.00 | Trail-exit close<VWAP |

### Cycle 25 — SELL (started 2025-11-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 11:05:00 | 1766.30 | 1775.07 | 0.00 | ORB-short ORB[1773.00,1785.00] vol=2.0x ATR=4.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 11:20:00 | 1760.29 | 1773.53 | 0.00 | T1 1.5R @ 1760.29 |
| Stop hit — per-position SL triggered | 2025-11-04 12:20:00 | 1766.30 | 1771.58 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-11-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 11:10:00 | 1742.30 | 1750.26 | 0.00 | ORB-short ORB[1750.80,1766.90] vol=1.5x ATR=4.72 |
| Stop hit — per-position SL triggered | 2025-11-11 11:35:00 | 1747.02 | 1749.86 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-11-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 09:45:00 | 1774.80 | 1771.75 | 0.00 | ORB-long ORB[1760.00,1772.80] vol=2.6x ATR=5.28 |
| Stop hit — per-position SL triggered | 2025-11-12 09:55:00 | 1769.52 | 1770.85 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-11-27 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 10:25:00 | 1661.60 | 1670.22 | 0.00 | ORB-short ORB[1670.40,1693.00] vol=1.5x ATR=5.87 |
| Stop hit — per-position SL triggered | 2025-11-27 10:35:00 | 1667.47 | 1669.33 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-11-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 11:05:00 | 1671.00 | 1665.93 | 0.00 | ORB-long ORB[1657.90,1669.60] vol=1.7x ATR=3.72 |
| Stop hit — per-position SL triggered | 2025-11-28 12:20:00 | 1667.28 | 1667.69 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-12-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 10:50:00 | 1636.50 | 1643.25 | 0.00 | ORB-short ORB[1646.80,1660.00] vol=3.4x ATR=4.84 |
| Stop hit — per-position SL triggered | 2025-12-03 11:45:00 | 1641.34 | 1641.51 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-12-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 11:10:00 | 1617.10 | 1601.01 | 0.00 | ORB-long ORB[1593.10,1611.90] vol=2.8x ATR=5.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 11:55:00 | 1625.68 | 1605.22 | 0.00 | T1 1.5R @ 1625.68 |
| Target hit | 2025-12-09 15:20:00 | 1633.40 | 1621.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — BUY (started 2025-12-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 10:10:00 | 1650.50 | 1635.51 | 0.00 | ORB-long ORB[1615.40,1633.50] vol=4.0x ATR=5.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 10:35:00 | 1658.49 | 1639.18 | 0.00 | T1 1.5R @ 1658.49 |
| Stop hit — per-position SL triggered | 2025-12-11 11:00:00 | 1650.50 | 1642.81 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-12-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 10:35:00 | 1624.40 | 1618.35 | 0.00 | ORB-long ORB[1600.20,1623.10] vol=3.9x ATR=4.83 |
| Stop hit — per-position SL triggered | 2025-12-19 10:45:00 | 1619.57 | 1618.76 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-12-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 09:50:00 | 1573.80 | 1581.26 | 0.00 | ORB-short ORB[1576.70,1592.00] vol=1.7x ATR=4.10 |
| Stop hit — per-position SL triggered | 2025-12-30 09:55:00 | 1577.90 | 1580.53 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-12-31 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:45:00 | 1591.30 | 1576.06 | 0.00 | ORB-long ORB[1565.00,1583.00] vol=1.5x ATR=4.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-31 11:00:00 | 1597.71 | 1578.58 | 0.00 | T1 1.5R @ 1597.71 |
| Stop hit — per-position SL triggered | 2025-12-31 11:05:00 | 1591.30 | 1579.20 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2026-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 09:30:00 | 1616.20 | 1612.33 | 0.00 | ORB-long ORB[1600.50,1616.00] vol=1.6x ATR=4.43 |
| Stop hit — per-position SL triggered | 2026-01-02 09:35:00 | 1611.77 | 1612.39 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2026-01-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 09:45:00 | 1480.80 | 1496.40 | 0.00 | ORB-short ORB[1499.40,1520.80] vol=1.5x ATR=6.49 |
| Stop hit — per-position SL triggered | 2026-01-14 09:50:00 | 1487.29 | 1494.98 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2026-02-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:40:00 | 1509.30 | 1515.98 | 0.00 | ORB-short ORB[1522.10,1538.40] vol=4.5x ATR=4.56 |
| Stop hit — per-position SL triggered | 2026-02-18 13:50:00 | 1513.86 | 1510.08 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2026-02-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:35:00 | 1503.10 | 1517.89 | 0.00 | ORB-short ORB[1518.00,1538.00] vol=1.8x ATR=4.34 |
| Stop hit — per-position SL triggered | 2026-02-19 10:40:00 | 1507.44 | 1517.15 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2026-02-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:45:00 | 1464.50 | 1469.27 | 0.00 | ORB-short ORB[1466.10,1483.10] vol=2.6x ATR=5.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:35:00 | 1456.73 | 1465.33 | 0.00 | T1 1.5R @ 1456.73 |
| Target hit | 2026-02-24 15:20:00 | 1439.00 | 1431.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 1354.30 | 1366.64 | 0.00 | ORB-short ORB[1357.50,1377.40] vol=2.1x ATR=4.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 11:00:00 | 1347.61 | 1364.48 | 0.00 | T1 1.5R @ 1347.61 |
| Stop hit — per-position SL triggered | 2026-03-06 11:35:00 | 1354.30 | 1358.43 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2026-03-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 10:35:00 | 1242.20 | 1249.23 | 0.00 | ORB-short ORB[1246.20,1263.70] vol=1.7x ATR=5.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 11:45:00 | 1233.42 | 1244.51 | 0.00 | T1 1.5R @ 1233.42 |
| Stop hit — per-position SL triggered | 2026-03-17 12:45:00 | 1242.20 | 1242.24 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2026-03-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 10:40:00 | 1280.90 | 1272.74 | 0.00 | ORB-long ORB[1257.30,1273.90] vol=1.7x ATR=5.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 10:45:00 | 1288.65 | 1275.54 | 0.00 | T1 1.5R @ 1288.65 |
| Target hit | 2026-03-18 15:20:00 | 1320.60 | 1300.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — BUY (started 2026-03-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:30:00 | 1300.20 | 1291.01 | 0.00 | ORB-long ORB[1278.90,1291.90] vol=1.7x ATR=6.31 |
| Stop hit — per-position SL triggered | 2026-03-20 09:55:00 | 1293.89 | 1293.74 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2026-04-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:45:00 | 1304.00 | 1310.44 | 0.00 | ORB-short ORB[1309.60,1320.80] vol=1.8x ATR=5.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 09:55:00 | 1295.64 | 1308.06 | 0.00 | T1 1.5R @ 1295.64 |
| Stop hit — per-position SL triggered | 2026-04-09 10:05:00 | 1304.00 | 1307.22 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2026-04-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:00:00 | 1355.00 | 1345.36 | 0.00 | ORB-long ORB[1334.40,1351.20] vol=1.9x ATR=5.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 14:45:00 | 1363.86 | 1353.84 | 0.00 | T1 1.5R @ 1363.86 |
| Stop hit — per-position SL triggered | 2026-04-17 15:00:00 | 1355.00 | 1353.99 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 1382.10 | 1388.72 | 0.00 | ORB-short ORB[1384.20,1399.40] vol=1.7x ATR=6.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:30:00 | 1373.08 | 1383.30 | 0.00 | T1 1.5R @ 1373.08 |
| Target hit | 2026-04-24 14:15:00 | 1367.00 | 1366.75 | 0.00 | Trail-exit close>VWAP |

### Cycle 48 — BUY (started 2026-04-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:55:00 | 1425.00 | 1415.27 | 0.00 | ORB-long ORB[1406.10,1421.80] vol=1.6x ATR=4.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:00:00 | 1432.38 | 1419.57 | 0.00 | T1 1.5R @ 1432.38 |
| Target hit | 2026-04-29 14:15:00 | 1435.00 | 1436.42 | 0.00 | Trail-exit close<VWAP |

### Cycle 49 — SELL (started 2026-04-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:25:00 | 1399.40 | 1412.25 | 0.00 | ORB-short ORB[1409.30,1430.00] vol=1.6x ATR=5.11 |
| Stop hit — per-position SL triggered | 2026-04-30 10:40:00 | 1404.51 | 1411.64 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2026-05-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:35:00 | 1501.60 | 1490.80 | 0.00 | ORB-long ORB[1480.00,1496.10] vol=2.2x ATR=5.37 |
| Stop hit — per-position SL triggered | 2026-05-08 11:00:00 | 1496.23 | 1492.80 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-08-11 10:50:00 | 1615.00 | 2025-08-11 11:55:00 | 1607.55 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-08-14 10:40:00 | 1613.40 | 2025-08-14 10:55:00 | 1617.78 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-08-18 09:40:00 | 1633.70 | 2025-08-18 10:00:00 | 1642.51 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-08-18 09:40:00 | 1633.70 | 2025-08-18 10:30:00 | 1633.80 | TARGET_HIT | 0.50 | 0.01% |
| SELL | retest1 | 2025-08-19 09:30:00 | 1608.70 | 2025-08-19 09:50:00 | 1613.91 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-08-25 10:15:00 | 1658.20 | 2025-08-25 12:10:00 | 1653.48 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-09-03 09:50:00 | 1574.50 | 2025-09-03 09:55:00 | 1569.07 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-09-05 10:20:00 | 1535.70 | 2025-09-05 10:30:00 | 1525.70 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2025-09-05 10:20:00 | 1535.70 | 2025-09-05 10:35:00 | 1535.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-11 09:50:00 | 1572.10 | 2025-09-11 10:45:00 | 1566.93 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-09-16 09:45:00 | 1611.00 | 2025-09-16 10:35:00 | 1606.07 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-09-17 10:15:00 | 1658.40 | 2025-09-17 10:30:00 | 1651.31 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-09-22 10:50:00 | 1638.80 | 2025-09-22 11:05:00 | 1633.76 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-09-29 10:20:00 | 1528.50 | 2025-09-29 10:30:00 | 1522.89 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-10-07 11:10:00 | 1530.70 | 2025-10-07 11:50:00 | 1534.43 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-10-10 09:30:00 | 1606.20 | 2025-10-10 09:35:00 | 1616.91 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2025-10-10 09:30:00 | 1606.20 | 2025-10-10 10:20:00 | 1608.20 | TARGET_HIT | 0.50 | 0.12% |
| BUY | retest1 | 2025-10-13 10:30:00 | 1629.50 | 2025-10-13 12:30:00 | 1623.43 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-10-17 09:35:00 | 1715.00 | 2025-10-17 09:40:00 | 1708.71 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-10-20 10:50:00 | 1723.10 | 2025-10-20 10:55:00 | 1730.87 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-10-20 10:50:00 | 1723.10 | 2025-10-20 12:10:00 | 1723.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-23 09:30:00 | 1732.10 | 2025-10-23 09:35:00 | 1726.56 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-10-24 09:35:00 | 1762.80 | 2025-10-24 09:50:00 | 1757.14 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-10-27 10:00:00 | 1790.30 | 2025-10-27 10:10:00 | 1784.48 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-10-29 10:45:00 | 1749.10 | 2025-10-29 11:00:00 | 1753.20 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-10-30 09:55:00 | 1746.20 | 2025-10-30 10:00:00 | 1751.76 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-10-31 09:30:00 | 1771.30 | 2025-10-31 09:40:00 | 1766.58 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-11-03 09:45:00 | 1766.60 | 2025-11-03 11:45:00 | 1777.02 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-11-03 09:45:00 | 1766.60 | 2025-11-03 14:10:00 | 1784.40 | TARGET_HIT | 0.50 | 1.01% |
| SELL | retest1 | 2025-11-04 11:05:00 | 1766.30 | 2025-11-04 11:20:00 | 1760.29 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-11-04 11:05:00 | 1766.30 | 2025-11-04 12:20:00 | 1766.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-11 11:10:00 | 1742.30 | 2025-11-11 11:35:00 | 1747.02 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-11-12 09:45:00 | 1774.80 | 2025-11-12 09:55:00 | 1769.52 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-11-27 10:25:00 | 1661.60 | 2025-11-27 10:35:00 | 1667.47 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-11-28 11:05:00 | 1671.00 | 2025-11-28 12:20:00 | 1667.28 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-12-03 10:50:00 | 1636.50 | 2025-12-03 11:45:00 | 1641.34 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-12-09 11:10:00 | 1617.10 | 2025-12-09 11:55:00 | 1625.68 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-12-09 11:10:00 | 1617.10 | 2025-12-09 15:20:00 | 1633.40 | TARGET_HIT | 0.50 | 1.01% |
| BUY | retest1 | 2025-12-11 10:10:00 | 1650.50 | 2025-12-11 10:35:00 | 1658.49 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-12-11 10:10:00 | 1650.50 | 2025-12-11 11:00:00 | 1650.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-19 10:35:00 | 1624.40 | 2025-12-19 10:45:00 | 1619.57 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-12-30 09:50:00 | 1573.80 | 2025-12-30 09:55:00 | 1577.90 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-12-31 10:45:00 | 1591.30 | 2025-12-31 11:00:00 | 1597.71 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-12-31 10:45:00 | 1591.30 | 2025-12-31 11:05:00 | 1591.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-02 09:30:00 | 1616.20 | 2026-01-02 09:35:00 | 1611.77 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-01-14 09:45:00 | 1480.80 | 2026-01-14 09:50:00 | 1487.29 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-02-18 10:40:00 | 1509.30 | 2026-02-18 13:50:00 | 1513.86 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-19 10:35:00 | 1503.10 | 2026-02-19 10:40:00 | 1507.44 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-24 09:45:00 | 1464.50 | 2026-02-24 11:35:00 | 1456.73 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-02-24 09:45:00 | 1464.50 | 2026-02-24 15:20:00 | 1439.00 | TARGET_HIT | 0.50 | 1.74% |
| SELL | retest1 | 2026-03-06 10:45:00 | 1354.30 | 2026-03-06 11:00:00 | 1347.61 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-03-06 10:45:00 | 1354.30 | 2026-03-06 11:35:00 | 1354.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-17 10:35:00 | 1242.20 | 2026-03-17 11:45:00 | 1233.42 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2026-03-17 10:35:00 | 1242.20 | 2026-03-17 12:45:00 | 1242.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-18 10:40:00 | 1280.90 | 2026-03-18 10:45:00 | 1288.65 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-03-18 10:40:00 | 1280.90 | 2026-03-18 15:20:00 | 1320.60 | TARGET_HIT | 0.50 | 3.10% |
| BUY | retest1 | 2026-03-20 09:30:00 | 1300.20 | 2026-03-20 09:55:00 | 1293.89 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-04-09 09:45:00 | 1304.00 | 2026-04-09 09:55:00 | 1295.64 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2026-04-09 09:45:00 | 1304.00 | 2026-04-09 10:05:00 | 1304.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 10:00:00 | 1355.00 | 2026-04-17 14:45:00 | 1363.86 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-04-17 10:00:00 | 1355.00 | 2026-04-17 15:00:00 | 1355.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-24 09:30:00 | 1382.10 | 2026-04-24 10:30:00 | 1373.08 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2026-04-24 09:30:00 | 1382.10 | 2026-04-24 14:15:00 | 1367.00 | TARGET_HIT | 0.50 | 1.09% |
| BUY | retest1 | 2026-04-29 09:55:00 | 1425.00 | 2026-04-29 10:00:00 | 1432.38 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-04-29 09:55:00 | 1425.00 | 2026-04-29 14:15:00 | 1435.00 | TARGET_HIT | 0.50 | 0.70% |
| SELL | retest1 | 2026-04-30 10:25:00 | 1399.40 | 2026-04-30 10:40:00 | 1404.51 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-05-08 10:35:00 | 1501.60 | 2026-05-08 11:00:00 | 1496.23 | STOP_HIT | 1.00 | -0.36% |
