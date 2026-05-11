# J.B. Chemicals & Pharmaceuticals Ltd. (JBCHEPHARM)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 2155.00
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
| ENTRY1 | 87 |
| ENTRY2 | 0 |
| PARTIAL | 34 |
| TARGET_HIT | 14 |
| STOP_HIT | 73 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 121 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 48 / 73
- **Target hits / Stop hits / Partials:** 14 / 73 / 34
- **Avg / median % per leg:** 0.13% / 0.00%
- **Sum % (uncompounded):** 15.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 65 | 24 | 36.9% | 7 | 41 | 17 | 0.15% | 9.8% |
| BUY @ 2nd Alert (retest1) | 65 | 24 | 36.9% | 7 | 41 | 17 | 0.15% | 9.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 56 | 24 | 42.9% | 7 | 32 | 17 | 0.10% | 5.8% |
| SELL @ 2nd Alert (retest1) | 56 | 24 | 42.9% | 7 | 32 | 17 | 0.10% | 5.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 121 | 48 | 39.7% | 14 | 73 | 34 | 0.13% | 15.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-12 10:25:00 | 1560.90 | 1552.05 | 0.00 | ORB-long ORB[1535.60,1551.90] vol=6.8x ATR=8.06 |
| Stop hit — per-position SL triggered | 2025-05-12 10:30:00 | 1552.84 | 1552.60 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-06-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-06 09:35:00 | 1690.70 | 1697.23 | 0.00 | ORB-short ORB[1692.60,1708.40] vol=1.5x ATR=4.68 |
| Stop hit — per-position SL triggered | 2025-06-06 09:50:00 | 1695.38 | 1695.75 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-06-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-09 09:30:00 | 1689.70 | 1698.00 | 0.00 | ORB-short ORB[1694.20,1713.20] vol=2.1x ATR=5.03 |
| Stop hit — per-position SL triggered | 2025-06-09 09:35:00 | 1694.73 | 1697.56 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-06-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 10:20:00 | 1694.90 | 1687.34 | 0.00 | ORB-long ORB[1671.00,1694.50] vol=2.2x ATR=4.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-11 10:25:00 | 1701.90 | 1689.05 | 0.00 | T1 1.5R @ 1701.90 |
| Stop hit — per-position SL triggered | 2025-06-11 10:35:00 | 1694.90 | 1690.40 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-12 09:30:00 | 1753.80 | 1742.47 | 0.00 | ORB-long ORB[1723.50,1742.00] vol=3.1x ATR=8.72 |
| Stop hit — per-position SL triggered | 2025-06-12 09:35:00 | 1745.08 | 1744.35 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 11:15:00 | 1692.10 | 1706.86 | 0.00 | ORB-short ORB[1710.80,1723.20] vol=2.1x ATR=4.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 1686.01 | 1702.38 | 0.00 | T1 1.5R @ 1686.01 |
| Target hit | 2025-06-19 15:20:00 | 1670.70 | 1687.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2025-06-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 10:45:00 | 1694.60 | 1686.06 | 0.00 | ORB-long ORB[1666.10,1689.60] vol=1.8x ATR=7.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 11:40:00 | 1705.10 | 1688.22 | 0.00 | T1 1.5R @ 1705.10 |
| Stop hit — per-position SL triggered | 2025-06-20 12:00:00 | 1694.60 | 1688.97 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 10:55:00 | 1778.60 | 1783.31 | 0.00 | ORB-short ORB[1778.80,1801.40] vol=4.0x ATR=5.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 12:15:00 | 1770.70 | 1780.66 | 0.00 | T1 1.5R @ 1770.70 |
| Target hit | 2025-06-26 15:20:00 | 1751.20 | 1766.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 11:15:00 | 1773.00 | 1759.84 | 0.00 | ORB-long ORB[1750.90,1769.00] vol=1.6x ATR=4.73 |
| Stop hit — per-position SL triggered | 2025-06-27 11:25:00 | 1768.27 | 1760.59 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-07-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-03 09:45:00 | 1632.20 | 1633.91 | 0.00 | ORB-short ORB[1632.50,1647.20] vol=4.2x ATR=3.47 |
| Stop hit — per-position SL triggered | 2025-07-03 10:00:00 | 1635.67 | 1634.04 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-07-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 11:10:00 | 1618.10 | 1625.34 | 0.00 | ORB-short ORB[1624.30,1635.50] vol=7.2x ATR=2.73 |
| Stop hit — per-position SL triggered | 2025-07-08 11:15:00 | 1620.83 | 1625.27 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-07-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 10:05:00 | 1616.60 | 1622.87 | 0.00 | ORB-short ORB[1622.20,1636.60] vol=1.6x ATR=3.26 |
| Stop hit — per-position SL triggered | 2025-07-10 10:30:00 | 1619.86 | 1621.92 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-07-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 10:45:00 | 1622.30 | 1617.81 | 0.00 | ORB-long ORB[1604.00,1618.80] vol=3.1x ATR=3.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 11:15:00 | 1627.46 | 1622.31 | 0.00 | T1 1.5R @ 1627.46 |
| Target hit | 2025-07-14 15:20:00 | 1639.00 | 1628.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2025-07-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 09:55:00 | 1662.00 | 1653.93 | 0.00 | ORB-long ORB[1642.00,1659.10] vol=1.6x ATR=4.17 |
| Stop hit — per-position SL triggered | 2025-07-16 10:10:00 | 1657.83 | 1654.41 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 10:20:00 | 1656.90 | 1652.97 | 0.00 | ORB-long ORB[1640.00,1654.10] vol=3.3x ATR=3.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 10:35:00 | 1661.74 | 1654.10 | 0.00 | T1 1.5R @ 1661.74 |
| Target hit | 2025-07-21 15:20:00 | 1685.70 | 1672.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2025-07-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 11:10:00 | 1670.10 | 1676.24 | 0.00 | ORB-short ORB[1671.90,1687.00] vol=2.0x ATR=4.08 |
| Stop hit — per-position SL triggered | 2025-07-22 11:25:00 | 1674.18 | 1675.86 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 09:40:00 | 1737.10 | 1730.82 | 0.00 | ORB-long ORB[1720.00,1736.80] vol=1.9x ATR=7.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 09:50:00 | 1747.83 | 1738.06 | 0.00 | T1 1.5R @ 1747.83 |
| Target hit | 2025-07-29 15:20:00 | 1803.70 | 1782.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — SELL (started 2025-07-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 11:05:00 | 1788.50 | 1797.16 | 0.00 | ORB-short ORB[1790.40,1810.80] vol=5.5x ATR=5.06 |
| Stop hit — per-position SL triggered | 2025-07-30 11:15:00 | 1793.56 | 1797.02 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-31 09:35:00 | 1755.00 | 1766.45 | 0.00 | ORB-short ORB[1755.70,1781.60] vol=2.5x ATR=6.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 10:10:00 | 1745.45 | 1759.18 | 0.00 | T1 1.5R @ 1745.45 |
| Stop hit — per-position SL triggered | 2025-07-31 12:10:00 | 1755.00 | 1756.44 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-04 11:15:00 | 1720.00 | 1721.34 | 0.00 | ORB-short ORB[1721.60,1740.10] vol=3.1x ATR=4.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-04 13:05:00 | 1713.59 | 1720.69 | 0.00 | T1 1.5R @ 1713.59 |
| Stop hit — per-position SL triggered | 2025-08-04 13:55:00 | 1720.00 | 1719.90 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-08-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 09:30:00 | 1694.80 | 1700.12 | 0.00 | ORB-short ORB[1697.00,1710.30] vol=2.1x ATR=4.96 |
| Stop hit — per-position SL triggered | 2025-08-05 09:55:00 | 1699.76 | 1698.57 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-08-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 11:05:00 | 1665.10 | 1676.77 | 0.00 | ORB-short ORB[1679.20,1702.90] vol=2.0x ATR=4.08 |
| Stop hit — per-position SL triggered | 2025-08-06 12:10:00 | 1669.18 | 1672.49 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-08-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 09:40:00 | 1698.40 | 1701.28 | 0.00 | ORB-short ORB[1699.80,1719.10] vol=3.0x ATR=5.57 |
| Stop hit — per-position SL triggered | 2025-08-12 09:55:00 | 1703.97 | 1701.22 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-08-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 09:50:00 | 1721.60 | 1714.93 | 0.00 | ORB-long ORB[1704.40,1718.30] vol=1.5x ATR=3.33 |
| Stop hit — per-position SL triggered | 2025-08-13 10:00:00 | 1718.27 | 1715.84 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-08-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 09:45:00 | 1708.80 | 1718.58 | 0.00 | ORB-short ORB[1716.10,1727.00] vol=1.8x ATR=3.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-14 10:00:00 | 1703.27 | 1716.74 | 0.00 | T1 1.5R @ 1703.27 |
| Stop hit — per-position SL triggered | 2025-08-14 10:15:00 | 1708.80 | 1712.86 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-08-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 10:55:00 | 1721.30 | 1718.40 | 0.00 | ORB-long ORB[1713.20,1721.10] vol=3.3x ATR=2.68 |
| Stop hit — per-position SL triggered | 2025-08-19 11:05:00 | 1718.62 | 1718.57 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-08-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 10:50:00 | 1736.50 | 1726.87 | 0.00 | ORB-long ORB[1720.20,1735.40] vol=2.0x ATR=3.38 |
| Stop hit — per-position SL triggered | 2025-08-21 11:15:00 | 1733.12 | 1728.20 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 11:15:00 | 1731.40 | 1729.71 | 0.00 | ORB-long ORB[1719.40,1730.00] vol=1.7x ATR=3.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 11:35:00 | 1736.53 | 1731.75 | 0.00 | T1 1.5R @ 1736.53 |
| Target hit | 2025-08-22 11:55:00 | 1732.00 | 1732.06 | 0.00 | Trail-exit close<VWAP |

### Cycle 29 — BUY (started 2025-08-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-29 11:00:00 | 1722.90 | 1713.84 | 0.00 | ORB-long ORB[1705.40,1719.40] vol=4.0x ATR=3.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 11:10:00 | 1727.91 | 1714.39 | 0.00 | T1 1.5R @ 1727.91 |
| Stop hit — per-position SL triggered | 2025-08-29 12:05:00 | 1722.90 | 1716.22 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-09-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 11:05:00 | 1721.00 | 1717.78 | 0.00 | ORB-long ORB[1710.00,1720.60] vol=2.9x ATR=3.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-03 11:15:00 | 1725.92 | 1718.86 | 0.00 | T1 1.5R @ 1725.92 |
| Stop hit — per-position SL triggered | 2025-09-03 11:30:00 | 1721.00 | 1720.36 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-09-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-05 09:45:00 | 1736.60 | 1724.78 | 0.00 | ORB-long ORB[1715.00,1728.10] vol=1.5x ATR=4.96 |
| Stop hit — per-position SL triggered | 2025-09-05 10:15:00 | 1731.64 | 1727.93 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-09-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 10:00:00 | 1701.00 | 1703.76 | 0.00 | ORB-short ORB[1702.80,1715.50] vol=3.5x ATR=3.04 |
| Stop hit — per-position SL triggered | 2025-09-12 11:05:00 | 1704.04 | 1702.84 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-09-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-15 11:10:00 | 1674.10 | 1682.71 | 0.00 | ORB-short ORB[1678.00,1700.10] vol=1.5x ATR=3.39 |
| Stop hit — per-position SL triggered | 2025-09-15 11:40:00 | 1677.49 | 1681.27 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-09-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 09:45:00 | 1675.70 | 1686.79 | 0.00 | ORB-short ORB[1683.20,1696.50] vol=1.7x ATR=4.22 |
| Stop hit — per-position SL triggered | 2025-09-17 09:55:00 | 1679.92 | 1685.05 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-09-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 09:35:00 | 1680.20 | 1684.94 | 0.00 | ORB-short ORB[1684.70,1698.90] vol=8.0x ATR=3.97 |
| Stop hit — per-position SL triggered | 2025-09-18 09:40:00 | 1684.17 | 1684.91 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-09-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 11:05:00 | 1704.30 | 1697.61 | 0.00 | ORB-long ORB[1685.90,1701.50] vol=4.8x ATR=3.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 11:15:00 | 1709.35 | 1698.51 | 0.00 | T1 1.5R @ 1709.35 |
| Stop hit — per-position SL triggered | 2025-09-19 11:40:00 | 1704.30 | 1700.17 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-09-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 10:20:00 | 1710.00 | 1713.59 | 0.00 | ORB-short ORB[1713.60,1734.40] vol=2.0x ATR=3.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 10:30:00 | 1704.34 | 1712.50 | 0.00 | T1 1.5R @ 1704.34 |
| Target hit | 2025-09-24 15:20:00 | 1696.00 | 1702.32 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — SELL (started 2025-09-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-25 10:00:00 | 1685.80 | 1693.69 | 0.00 | ORB-short ORB[1689.10,1709.10] vol=1.5x ATR=4.09 |
| Stop hit — per-position SL triggered | 2025-09-25 10:05:00 | 1689.89 | 1693.40 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-09-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-26 09:55:00 | 1663.50 | 1653.20 | 0.00 | ORB-long ORB[1643.60,1662.40] vol=1.7x ATR=7.28 |
| Stop hit — per-position SL triggered | 2025-09-26 10:30:00 | 1656.22 | 1656.04 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-09-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 09:35:00 | 1694.00 | 1688.01 | 0.00 | ORB-long ORB[1670.90,1692.80] vol=2.6x ATR=5.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 11:20:00 | 1702.71 | 1693.03 | 0.00 | T1 1.5R @ 1702.71 |
| Stop hit — per-position SL triggered | 2025-09-29 11:35:00 | 1694.00 | 1693.51 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-09-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-30 11:00:00 | 1684.90 | 1690.92 | 0.00 | ORB-short ORB[1695.60,1716.00] vol=1.8x ATR=4.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 11:35:00 | 1678.66 | 1689.14 | 0.00 | T1 1.5R @ 1678.66 |
| Stop hit — per-position SL triggered | 2025-09-30 12:10:00 | 1684.90 | 1685.78 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-10-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-09 10:30:00 | 1671.50 | 1677.15 | 0.00 | ORB-short ORB[1677.10,1687.10] vol=3.1x ATR=4.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 10:45:00 | 1664.39 | 1672.75 | 0.00 | T1 1.5R @ 1664.39 |
| Target hit | 2025-10-09 11:45:00 | 1668.80 | 1668.35 | 0.00 | Trail-exit close>VWAP |

### Cycle 43 — BUY (started 2025-10-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:00:00 | 1676.50 | 1667.13 | 0.00 | ORB-long ORB[1649.00,1668.40] vol=1.7x ATR=4.34 |
| Stop hit — per-position SL triggered | 2025-10-15 10:05:00 | 1672.16 | 1667.81 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-10-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 10:30:00 | 1681.90 | 1676.71 | 0.00 | ORB-long ORB[1662.00,1670.30] vol=9.3x ATR=3.42 |
| Stop hit — per-position SL triggered | 2025-10-17 10:40:00 | 1678.48 | 1678.03 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-10-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 10:50:00 | 1698.70 | 1701.00 | 0.00 | ORB-short ORB[1699.00,1705.70] vol=1.6x ATR=2.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 11:00:00 | 1695.33 | 1700.82 | 0.00 | T1 1.5R @ 1695.33 |
| Stop hit — per-position SL triggered | 2025-10-28 11:05:00 | 1698.70 | 1700.78 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-10-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:55:00 | 1704.70 | 1694.14 | 0.00 | ORB-long ORB[1683.00,1697.40] vol=3.7x ATR=2.91 |
| Stop hit — per-position SL triggered | 2025-10-29 11:05:00 | 1701.79 | 1696.01 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-11-03 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 10:30:00 | 1700.10 | 1699.05 | 0.00 | ORB-long ORB[1683.30,1692.90] vol=12.8x ATR=3.77 |
| Stop hit — per-position SL triggered | 2025-11-03 10:50:00 | 1696.33 | 1699.11 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-11-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-07 10:30:00 | 1692.70 | 1686.31 | 0.00 | ORB-long ORB[1675.00,1691.00] vol=1.5x ATR=3.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 10:35:00 | 1698.59 | 1687.71 | 0.00 | T1 1.5R @ 1698.59 |
| Stop hit — per-position SL triggered | 2025-11-07 10:55:00 | 1692.70 | 1690.05 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-11 10:15:00 | 1840.70 | 1826.59 | 0.00 | ORB-long ORB[1812.10,1829.80] vol=1.8x ATR=6.77 |
| Stop hit — per-position SL triggered | 2025-11-11 11:15:00 | 1833.93 | 1833.23 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-11-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-12 10:55:00 | 1811.40 | 1816.20 | 0.00 | ORB-short ORB[1815.00,1840.00] vol=5.6x ATR=4.68 |
| Stop hit — per-position SL triggered | 2025-11-12 11:00:00 | 1816.08 | 1816.14 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-11-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 10:20:00 | 1838.20 | 1827.85 | 0.00 | ORB-long ORB[1805.80,1830.30] vol=1.9x ATR=4.27 |
| Stop hit — per-position SL triggered | 2025-11-13 11:10:00 | 1833.93 | 1833.49 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-17 11:15:00 | 1815.20 | 1825.23 | 0.00 | ORB-short ORB[1821.10,1834.80] vol=2.1x ATR=3.51 |
| Stop hit — per-position SL triggered | 2025-11-17 11:45:00 | 1818.71 | 1824.25 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-11-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 09:30:00 | 1778.20 | 1783.38 | 0.00 | ORB-short ORB[1783.00,1800.00] vol=3.9x ATR=4.56 |
| Stop hit — per-position SL triggered | 2025-11-19 09:40:00 | 1782.76 | 1782.81 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-11-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:05:00 | 1732.70 | 1737.23 | 0.00 | ORB-short ORB[1738.20,1748.40] vol=4.7x ATR=3.81 |
| Stop hit — per-position SL triggered | 2025-11-21 10:10:00 | 1736.51 | 1736.82 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-24 10:15:00 | 1750.90 | 1737.48 | 0.00 | ORB-long ORB[1725.30,1737.60] vol=1.6x ATR=4.06 |
| Stop hit — per-position SL triggered | 2025-11-24 10:30:00 | 1746.84 | 1743.14 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-11-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 09:50:00 | 1776.50 | 1772.05 | 0.00 | ORB-long ORB[1765.30,1771.90] vol=2.4x ATR=3.48 |
| Stop hit — per-position SL triggered | 2025-11-28 09:55:00 | 1773.02 | 1772.33 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-12-02 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 10:30:00 | 1759.70 | 1762.49 | 0.00 | ORB-short ORB[1764.20,1774.30] vol=1.6x ATR=3.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 10:40:00 | 1754.71 | 1761.83 | 0.00 | T1 1.5R @ 1754.71 |
| Target hit | 2025-12-02 13:20:00 | 1757.30 | 1756.50 | 0.00 | Trail-exit close>VWAP |

### Cycle 58 — BUY (started 2025-12-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 09:30:00 | 1770.80 | 1766.73 | 0.00 | ORB-long ORB[1757.50,1768.40] vol=2.8x ATR=4.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 10:05:00 | 1777.50 | 1772.95 | 0.00 | T1 1.5R @ 1777.50 |
| Target hit | 2025-12-04 15:20:00 | 1826.80 | 1807.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — SELL (started 2025-12-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 11:05:00 | 1789.40 | 1792.12 | 0.00 | ORB-short ORB[1790.90,1812.60] vol=2.4x ATR=4.56 |
| Stop hit — per-position SL triggered | 2025-12-09 11:20:00 | 1793.96 | 1792.08 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-12-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 09:30:00 | 1782.60 | 1786.18 | 0.00 | ORB-short ORB[1785.60,1799.90] vol=2.5x ATR=4.40 |
| Stop hit — per-position SL triggered | 2025-12-15 09:45:00 | 1787.00 | 1786.11 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-12-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 11:05:00 | 1801.60 | 1807.27 | 0.00 | ORB-short ORB[1804.00,1822.20] vol=1.9x ATR=3.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 12:30:00 | 1796.71 | 1804.57 | 0.00 | T1 1.5R @ 1796.71 |
| Stop hit — per-position SL triggered | 2025-12-29 13:10:00 | 1801.60 | 1803.78 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-12-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 11:10:00 | 1788.30 | 1792.93 | 0.00 | ORB-short ORB[1788.60,1803.40] vol=5.0x ATR=4.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 11:55:00 | 1781.91 | 1790.70 | 0.00 | T1 1.5R @ 1781.91 |
| Stop hit — per-position SL triggered | 2025-12-30 12:10:00 | 1788.30 | 1790.15 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2026-01-02 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:25:00 | 1829.60 | 1821.06 | 0.00 | ORB-long ORB[1806.10,1820.60] vol=3.0x ATR=3.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 11:10:00 | 1834.33 | 1826.05 | 0.00 | T1 1.5R @ 1834.33 |
| Target hit | 2026-01-02 15:20:00 | 1851.10 | 1841.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — BUY (started 2026-01-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 09:30:00 | 1900.70 | 1891.92 | 0.00 | ORB-long ORB[1876.40,1894.00] vol=2.4x ATR=5.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-07 09:55:00 | 1909.43 | 1898.05 | 0.00 | T1 1.5R @ 1909.43 |
| Stop hit — per-position SL triggered | 2026-01-07 10:45:00 | 1900.70 | 1902.24 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-01-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 11:10:00 | 1873.00 | 1864.33 | 0.00 | ORB-long ORB[1855.20,1871.80] vol=1.7x ATR=3.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 11:20:00 | 1878.32 | 1866.98 | 0.00 | T1 1.5R @ 1878.32 |
| Target hit | 2026-01-14 15:20:00 | 1897.70 | 1876.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 66 — BUY (started 2026-01-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-22 11:05:00 | 1903.80 | 1900.10 | 0.00 | ORB-long ORB[1882.00,1902.40] vol=1.7x ATR=5.18 |
| Stop hit — per-position SL triggered | 2026-01-22 11:10:00 | 1898.62 | 1900.09 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-01-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-27 09:35:00 | 1879.40 | 1858.34 | 0.00 | ORB-long ORB[1839.60,1862.40] vol=2.1x ATR=7.68 |
| Stop hit — per-position SL triggered | 2026-01-27 09:40:00 | 1871.72 | 1859.37 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2026-02-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 10:55:00 | 1883.50 | 1865.58 | 0.00 | ORB-long ORB[1853.20,1877.20] vol=2.0x ATR=5.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 11:10:00 | 1891.41 | 1868.54 | 0.00 | T1 1.5R @ 1891.41 |
| Stop hit — per-position SL triggered | 2026-02-01 11:20:00 | 1883.50 | 1869.86 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2026-02-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:00:00 | 1891.10 | 1871.35 | 0.00 | ORB-long ORB[1848.70,1865.70] vol=2.7x ATR=4.34 |
| Stop hit — per-position SL triggered | 2026-02-09 11:25:00 | 1886.76 | 1874.45 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2026-02-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 11:00:00 | 1909.50 | 1900.18 | 0.00 | ORB-long ORB[1883.50,1902.60] vol=2.6x ATR=4.45 |
| Stop hit — per-position SL triggered | 2026-02-10 11:50:00 | 1905.05 | 1904.71 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-02-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:00:00 | 1978.10 | 1988.11 | 0.00 | ORB-short ORB[1982.00,1997.50] vol=2.2x ATR=4.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:20:00 | 1971.04 | 1984.25 | 0.00 | T1 1.5R @ 1971.04 |
| Stop hit — per-position SL triggered | 2026-02-18 11:45:00 | 1978.10 | 1980.28 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2026-02-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 09:45:00 | 2033.10 | 2019.65 | 0.00 | ORB-long ORB[2004.00,2023.00] vol=2.2x ATR=6.45 |
| Stop hit — per-position SL triggered | 2026-02-23 09:50:00 | 2026.65 | 2020.93 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-02-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:05:00 | 2106.00 | 2097.27 | 0.00 | ORB-long ORB[2072.00,2092.90] vol=1.9x ATR=6.26 |
| Stop hit — per-position SL triggered | 2026-02-26 10:20:00 | 2099.74 | 2097.91 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-02-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:55:00 | 2071.90 | 2079.14 | 0.00 | ORB-short ORB[2072.00,2092.60] vol=1.7x ATR=3.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:25:00 | 2066.27 | 2075.34 | 0.00 | T1 1.5R @ 2066.27 |
| Target hit | 2026-02-27 15:20:00 | 2052.40 | 2063.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — BUY (started 2026-03-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 09:30:00 | 2068.40 | 2058.46 | 0.00 | ORB-long ORB[2048.20,2066.00] vol=2.3x ATR=7.69 |
| Stop hit — per-position SL triggered | 2026-03-05 10:25:00 | 2060.71 | 2065.58 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 2044.10 | 2058.63 | 0.00 | ORB-short ORB[2066.40,2080.00] vol=2.7x ATR=5.17 |
| Stop hit — per-position SL triggered | 2026-03-06 11:15:00 | 2049.27 | 2054.48 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-03-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:20:00 | 2105.10 | 2099.77 | 0.00 | ORB-long ORB[2077.20,2104.80] vol=1.9x ATR=6.27 |
| Stop hit — per-position SL triggered | 2026-03-10 10:50:00 | 2098.83 | 2100.91 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-03-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:10:00 | 2113.20 | 2123.86 | 0.00 | ORB-short ORB[2118.30,2139.20] vol=1.5x ATR=6.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:40:00 | 2104.02 | 2118.56 | 0.00 | T1 1.5R @ 2104.02 |
| Target hit | 2026-03-13 11:35:00 | 2112.40 | 2110.23 | 0.00 | Trail-exit close>VWAP |

### Cycle 79 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:15:00 | 2112.00 | 2102.99 | 0.00 | ORB-long ORB[2083.10,2099.00] vol=6.9x ATR=7.09 |
| Stop hit — per-position SL triggered | 2026-03-17 10:25:00 | 2104.91 | 2103.25 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2026-03-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-18 10:00:00 | 2092.50 | 2104.64 | 0.00 | ORB-short ORB[2100.10,2126.50] vol=1.6x ATR=7.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 11:15:00 | 2081.07 | 2100.04 | 0.00 | T1 1.5R @ 2081.07 |
| Stop hit — per-position SL triggered | 2026-03-18 11:40:00 | 2092.50 | 2099.00 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2026-03-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:55:00 | 2096.90 | 2090.51 | 0.00 | ORB-long ORB[2063.10,2092.00] vol=3.1x ATR=5.52 |
| Stop hit — per-position SL triggered | 2026-03-25 11:25:00 | 2091.38 | 2091.55 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2026-03-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-30 11:00:00 | 2064.80 | 2060.03 | 0.00 | ORB-long ORB[2030.20,2055.00] vol=1.6x ATR=6.82 |
| Stop hit — per-position SL triggered | 2026-03-30 11:10:00 | 2057.98 | 2060.05 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:15:00 | 1972.60 | 1976.67 | 0.00 | ORB-short ORB[1975.20,1999.80] vol=2.4x ATR=4.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 10:35:00 | 1966.33 | 1976.36 | 0.00 | T1 1.5R @ 1966.33 |
| Stop hit — per-position SL triggered | 2026-04-16 11:50:00 | 1972.60 | 1974.31 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2026-04-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:40:00 | 2041.40 | 2025.30 | 0.00 | ORB-long ORB[2008.20,2031.80] vol=1.8x ATR=6.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:45:00 | 2051.34 | 2028.83 | 0.00 | T1 1.5R @ 2051.34 |
| Stop hit — per-position SL triggered | 2026-04-27 09:55:00 | 2041.40 | 2030.06 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2026-04-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:40:00 | 2073.00 | 2066.66 | 0.00 | ORB-long ORB[2040.00,2068.00] vol=1.6x ATR=7.43 |
| Stop hit — per-position SL triggered | 2026-04-29 09:50:00 | 2065.57 | 2067.80 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2026-05-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:40:00 | 2089.00 | 2077.46 | 0.00 | ORB-long ORB[2064.00,2085.40] vol=2.2x ATR=7.47 |
| Stop hit — per-position SL triggered | 2026-05-04 10:00:00 | 2081.53 | 2079.67 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2026-05-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:40:00 | 2120.00 | 2110.93 | 0.00 | ORB-long ORB[2100.00,2116.70] vol=1.9x ATR=6.36 |
| Stop hit — per-position SL triggered | 2026-05-06 09:45:00 | 2113.64 | 2113.96 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-12 10:25:00 | 1560.90 | 2025-05-12 10:30:00 | 1552.84 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2025-06-06 09:35:00 | 1690.70 | 2025-06-06 09:50:00 | 1695.38 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-06-09 09:30:00 | 1689.70 | 2025-06-09 09:35:00 | 1694.73 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-06-11 10:20:00 | 1694.90 | 2025-06-11 10:25:00 | 1701.90 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-06-11 10:20:00 | 1694.90 | 2025-06-11 10:35:00 | 1694.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-12 09:30:00 | 1753.80 | 2025-06-12 09:35:00 | 1745.08 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2025-06-19 11:15:00 | 1692.10 | 2025-06-19 12:15:00 | 1686.01 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-06-19 11:15:00 | 1692.10 | 2025-06-19 15:20:00 | 1670.70 | TARGET_HIT | 0.50 | 1.26% |
| BUY | retest1 | 2025-06-20 10:45:00 | 1694.60 | 2025-06-20 11:40:00 | 1705.10 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-06-20 10:45:00 | 1694.60 | 2025-06-20 12:00:00 | 1694.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-26 10:55:00 | 1778.60 | 2025-06-26 12:15:00 | 1770.70 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-06-26 10:55:00 | 1778.60 | 2025-06-26 15:20:00 | 1751.20 | TARGET_HIT | 0.50 | 1.54% |
| BUY | retest1 | 2025-06-27 11:15:00 | 1773.00 | 2025-06-27 11:25:00 | 1768.27 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-07-03 09:45:00 | 1632.20 | 2025-07-03 10:00:00 | 1635.67 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-07-08 11:10:00 | 1618.10 | 2025-07-08 11:15:00 | 1620.83 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-07-10 10:05:00 | 1616.60 | 2025-07-10 10:30:00 | 1619.86 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-07-14 10:45:00 | 1622.30 | 2025-07-14 11:15:00 | 1627.46 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-07-14 10:45:00 | 1622.30 | 2025-07-14 15:20:00 | 1639.00 | TARGET_HIT | 0.50 | 1.03% |
| BUY | retest1 | 2025-07-16 09:55:00 | 1662.00 | 2025-07-16 10:10:00 | 1657.83 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-07-21 10:20:00 | 1656.90 | 2025-07-21 10:35:00 | 1661.74 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-07-21 10:20:00 | 1656.90 | 2025-07-21 15:20:00 | 1685.70 | TARGET_HIT | 0.50 | 1.74% |
| SELL | retest1 | 2025-07-22 11:10:00 | 1670.10 | 2025-07-22 11:25:00 | 1674.18 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-07-29 09:40:00 | 1737.10 | 2025-07-29 09:50:00 | 1747.83 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-07-29 09:40:00 | 1737.10 | 2025-07-29 15:20:00 | 1803.70 | TARGET_HIT | 0.50 | 3.83% |
| SELL | retest1 | 2025-07-30 11:05:00 | 1788.50 | 2025-07-30 11:15:00 | 1793.56 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-07-31 09:35:00 | 1755.00 | 2025-07-31 10:10:00 | 1745.45 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-07-31 09:35:00 | 1755.00 | 2025-07-31 12:10:00 | 1755.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-04 11:15:00 | 1720.00 | 2025-08-04 13:05:00 | 1713.59 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-08-04 11:15:00 | 1720.00 | 2025-08-04 13:55:00 | 1720.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-05 09:30:00 | 1694.80 | 2025-08-05 09:55:00 | 1699.76 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-08-06 11:05:00 | 1665.10 | 2025-08-06 12:10:00 | 1669.18 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-08-12 09:40:00 | 1698.40 | 2025-08-12 09:55:00 | 1703.97 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-08-13 09:50:00 | 1721.60 | 2025-08-13 10:00:00 | 1718.27 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-08-14 09:45:00 | 1708.80 | 2025-08-14 10:00:00 | 1703.27 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-08-14 09:45:00 | 1708.80 | 2025-08-14 10:15:00 | 1708.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-19 10:55:00 | 1721.30 | 2025-08-19 11:05:00 | 1718.62 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-08-21 10:50:00 | 1736.50 | 2025-08-21 11:15:00 | 1733.12 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-08-22 11:15:00 | 1731.40 | 2025-08-22 11:35:00 | 1736.53 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-08-22 11:15:00 | 1731.40 | 2025-08-22 11:55:00 | 1732.00 | TARGET_HIT | 0.50 | 0.03% |
| BUY | retest1 | 2025-08-29 11:00:00 | 1722.90 | 2025-08-29 11:10:00 | 1727.91 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-08-29 11:00:00 | 1722.90 | 2025-08-29 12:05:00 | 1722.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-03 11:05:00 | 1721.00 | 2025-09-03 11:15:00 | 1725.92 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-09-03 11:05:00 | 1721.00 | 2025-09-03 11:30:00 | 1721.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-05 09:45:00 | 1736.60 | 2025-09-05 10:15:00 | 1731.64 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-09-12 10:00:00 | 1701.00 | 2025-09-12 11:05:00 | 1704.04 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-09-15 11:10:00 | 1674.10 | 2025-09-15 11:40:00 | 1677.49 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-09-17 09:45:00 | 1675.70 | 2025-09-17 09:55:00 | 1679.92 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-09-18 09:35:00 | 1680.20 | 2025-09-18 09:40:00 | 1684.17 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-09-19 11:05:00 | 1704.30 | 2025-09-19 11:15:00 | 1709.35 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-09-19 11:05:00 | 1704.30 | 2025-09-19 11:40:00 | 1704.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-24 10:20:00 | 1710.00 | 2025-09-24 10:30:00 | 1704.34 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-09-24 10:20:00 | 1710.00 | 2025-09-24 15:20:00 | 1696.00 | TARGET_HIT | 0.50 | 0.82% |
| SELL | retest1 | 2025-09-25 10:00:00 | 1685.80 | 2025-09-25 10:05:00 | 1689.89 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-09-26 09:55:00 | 1663.50 | 2025-09-26 10:30:00 | 1656.22 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-09-29 09:35:00 | 1694.00 | 2025-09-29 11:20:00 | 1702.71 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-09-29 09:35:00 | 1694.00 | 2025-09-29 11:35:00 | 1694.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-30 11:00:00 | 1684.90 | 2025-09-30 11:35:00 | 1678.66 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-09-30 11:00:00 | 1684.90 | 2025-09-30 12:10:00 | 1684.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-09 10:30:00 | 1671.50 | 2025-10-09 10:45:00 | 1664.39 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-10-09 10:30:00 | 1671.50 | 2025-10-09 11:45:00 | 1668.80 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2025-10-15 10:00:00 | 1676.50 | 2025-10-15 10:05:00 | 1672.16 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-10-17 10:30:00 | 1681.90 | 2025-10-17 10:40:00 | 1678.48 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-10-28 10:50:00 | 1698.70 | 2025-10-28 11:00:00 | 1695.33 | PARTIAL | 0.50 | 0.20% |
| SELL | retest1 | 2025-10-28 10:50:00 | 1698.70 | 2025-10-28 11:05:00 | 1698.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-29 10:55:00 | 1704.70 | 2025-10-29 11:05:00 | 1701.79 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-11-03 10:30:00 | 1700.10 | 2025-11-03 10:50:00 | 1696.33 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-11-07 10:30:00 | 1692.70 | 2025-11-07 10:35:00 | 1698.59 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-11-07 10:30:00 | 1692.70 | 2025-11-07 10:55:00 | 1692.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-11 10:15:00 | 1840.70 | 2025-11-11 11:15:00 | 1833.93 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-11-12 10:55:00 | 1811.40 | 2025-11-12 11:00:00 | 1816.08 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-11-13 10:20:00 | 1838.20 | 2025-11-13 11:10:00 | 1833.93 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-11-17 11:15:00 | 1815.20 | 2025-11-17 11:45:00 | 1818.71 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-11-19 09:30:00 | 1778.20 | 2025-11-19 09:40:00 | 1782.76 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-11-21 10:05:00 | 1732.70 | 2025-11-21 10:10:00 | 1736.51 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-11-24 10:15:00 | 1750.90 | 2025-11-24 10:30:00 | 1746.84 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-11-28 09:50:00 | 1776.50 | 2025-11-28 09:55:00 | 1773.02 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-12-02 10:30:00 | 1759.70 | 2025-12-02 10:40:00 | 1754.71 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-12-02 10:30:00 | 1759.70 | 2025-12-02 13:20:00 | 1757.30 | TARGET_HIT | 0.50 | 0.14% |
| BUY | retest1 | 2025-12-04 09:30:00 | 1770.80 | 2025-12-04 10:05:00 | 1777.50 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-12-04 09:30:00 | 1770.80 | 2025-12-04 15:20:00 | 1826.80 | TARGET_HIT | 0.50 | 3.16% |
| SELL | retest1 | 2025-12-09 11:05:00 | 1789.40 | 2025-12-09 11:20:00 | 1793.96 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-12-15 09:30:00 | 1782.60 | 2025-12-15 09:45:00 | 1787.00 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-12-29 11:05:00 | 1801.60 | 2025-12-29 12:30:00 | 1796.71 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-12-29 11:05:00 | 1801.60 | 2025-12-29 13:10:00 | 1801.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-30 11:10:00 | 1788.30 | 2025-12-30 11:55:00 | 1781.91 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-12-30 11:10:00 | 1788.30 | 2025-12-30 12:10:00 | 1788.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-02 10:25:00 | 1829.60 | 2026-01-02 11:10:00 | 1834.33 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2026-01-02 10:25:00 | 1829.60 | 2026-01-02 15:20:00 | 1851.10 | TARGET_HIT | 0.50 | 1.18% |
| BUY | retest1 | 2026-01-07 09:30:00 | 1900.70 | 2026-01-07 09:55:00 | 1909.43 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-01-07 09:30:00 | 1900.70 | 2026-01-07 10:45:00 | 1900.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-14 11:10:00 | 1873.00 | 2026-01-14 11:20:00 | 1878.32 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2026-01-14 11:10:00 | 1873.00 | 2026-01-14 15:20:00 | 1897.70 | TARGET_HIT | 0.50 | 1.32% |
| BUY | retest1 | 2026-01-22 11:05:00 | 1903.80 | 2026-01-22 11:10:00 | 1898.62 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-01-27 09:35:00 | 1879.40 | 2026-01-27 09:40:00 | 1871.72 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-02-01 10:55:00 | 1883.50 | 2026-02-01 11:10:00 | 1891.41 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-02-01 10:55:00 | 1883.50 | 2026-02-01 11:20:00 | 1883.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-09 11:00:00 | 1891.10 | 2026-02-09 11:25:00 | 1886.76 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-02-10 11:00:00 | 1909.50 | 2026-02-10 11:50:00 | 1905.05 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-02-18 11:00:00 | 1978.10 | 2026-02-18 11:20:00 | 1971.04 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-02-18 11:00:00 | 1978.10 | 2026-02-18 11:45:00 | 1978.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-23 09:45:00 | 2033.10 | 2026-02-23 09:50:00 | 2026.65 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-26 10:05:00 | 2106.00 | 2026-02-26 10:20:00 | 2099.74 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-27 10:55:00 | 2071.90 | 2026-02-27 11:25:00 | 2066.27 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2026-02-27 10:55:00 | 2071.90 | 2026-02-27 15:20:00 | 2052.40 | TARGET_HIT | 0.50 | 0.94% |
| BUY | retest1 | 2026-03-05 09:30:00 | 2068.40 | 2026-03-05 10:25:00 | 2060.71 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-03-06 10:45:00 | 2044.10 | 2026-03-06 11:15:00 | 2049.27 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-03-10 10:20:00 | 2105.10 | 2026-03-10 10:50:00 | 2098.83 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-13 10:10:00 | 2113.20 | 2026-03-13 10:40:00 | 2104.02 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-03-13 10:10:00 | 2113.20 | 2026-03-13 11:35:00 | 2112.40 | TARGET_HIT | 0.50 | 0.04% |
| BUY | retest1 | 2026-03-17 10:15:00 | 2112.00 | 2026-03-17 10:25:00 | 2104.91 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-18 10:00:00 | 2092.50 | 2026-03-18 11:15:00 | 2081.07 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-03-18 10:00:00 | 2092.50 | 2026-03-18 11:40:00 | 2092.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-25 10:55:00 | 2096.90 | 2026-03-25 11:25:00 | 2091.38 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-03-30 11:00:00 | 2064.80 | 2026-03-30 11:10:00 | 2057.98 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-04-16 10:15:00 | 1972.60 | 2026-04-16 10:35:00 | 1966.33 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-04-16 10:15:00 | 1972.60 | 2026-04-16 11:50:00 | 1972.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 09:40:00 | 2041.40 | 2026-04-27 09:45:00 | 2051.34 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-04-27 09:40:00 | 2041.40 | 2026-04-27 09:55:00 | 2041.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 09:40:00 | 2073.00 | 2026-04-29 09:50:00 | 2065.57 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-05-04 09:40:00 | 2089.00 | 2026-05-04 10:00:00 | 2081.53 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-05-06 09:40:00 | 2120.00 | 2026-05-06 09:45:00 | 2113.64 | STOP_HIT | 1.00 | -0.30% |
