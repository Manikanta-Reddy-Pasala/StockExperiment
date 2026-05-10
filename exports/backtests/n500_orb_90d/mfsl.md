# Max Financial Services Ltd. (MFSL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1695.00
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
| ENTRY1 | 21 |
| ENTRY2 | 0 |
| PARTIAL | 9 |
| TARGET_HIT | 3 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 30 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 18
- **Target hits / Stop hits / Partials:** 3 / 18 / 9
- **Avg / median % per leg:** 0.06% / 0.00%
- **Sum % (uncompounded):** 1.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 2 | 18.2% | 0 | 9 | 2 | -0.08% | -0.9% |
| BUY @ 2nd Alert (retest1) | 11 | 2 | 18.2% | 0 | 9 | 2 | -0.08% | -0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 19 | 10 | 52.6% | 3 | 9 | 7 | 0.15% | 2.8% |
| SELL @ 2nd Alert (retest1) | 19 | 10 | 52.6% | 3 | 9 | 7 | 0.15% | 2.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 30 | 12 | 40.0% | 3 | 18 | 9 | 0.06% | 1.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:45:00 | 1728.30 | 1721.84 | 0.00 | ORB-long ORB[1707.00,1727.40] vol=2.2x ATR=5.69 |
| Stop hit — per-position SL triggered | 2026-02-09 11:20:00 | 1722.61 | 1723.19 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 10:50:00 | 1739.90 | 1743.41 | 0.00 | ORB-short ORB[1740.20,1755.80] vol=5.0x ATR=4.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 11:10:00 | 1733.59 | 1742.14 | 0.00 | T1 1.5R @ 1733.59 |
| Stop hit — per-position SL triggered | 2026-02-11 11:20:00 | 1739.90 | 1741.90 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:45:00 | 1851.60 | 1838.66 | 0.00 | ORB-long ORB[1815.90,1837.50] vol=1.5x ATR=4.46 |
| Stop hit — per-position SL triggered | 2026-02-16 10:50:00 | 1847.14 | 1839.43 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 11:10:00 | 1845.80 | 1834.39 | 0.00 | ORB-long ORB[1810.30,1834.00] vol=2.4x ATR=4.28 |
| Stop hit — per-position SL triggered | 2026-02-20 11:15:00 | 1841.52 | 1835.84 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-04 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 10:40:00 | 1736.40 | 1759.43 | 0.00 | ORB-short ORB[1763.40,1784.90] vol=1.8x ATR=6.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 12:15:00 | 1727.17 | 1747.83 | 0.00 | T1 1.5R @ 1727.17 |
| Stop hit — per-position SL triggered | 2026-03-04 13:50:00 | 1736.40 | 1742.34 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:15:00 | 1738.50 | 1754.66 | 0.00 | ORB-short ORB[1753.00,1773.50] vol=2.2x ATR=4.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:50:00 | 1731.32 | 1751.33 | 0.00 | T1 1.5R @ 1731.32 |
| Stop hit — per-position SL triggered | 2026-03-05 14:50:00 | 1738.50 | 1739.36 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:35:00 | 1717.70 | 1729.75 | 0.00 | ORB-short ORB[1725.00,1748.50] vol=1.5x ATR=6.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 11:50:00 | 1708.12 | 1720.76 | 0.00 | T1 1.5R @ 1708.12 |
| Target hit | 2026-03-06 14:25:00 | 1709.90 | 1709.79 | 0.00 | Trail-exit close>VWAP |

### Cycle 8 — SELL (started 2026-03-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 10:55:00 | 1720.80 | 1723.65 | 0.00 | ORB-short ORB[1721.10,1741.20] vol=1.9x ATR=5.80 |
| Stop hit — per-position SL triggered | 2026-03-10 13:35:00 | 1726.60 | 1722.05 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 11:00:00 | 1747.20 | 1734.36 | 0.00 | ORB-long ORB[1709.40,1733.20] vol=5.9x ATR=5.07 |
| Stop hit — per-position SL triggered | 2026-03-11 11:10:00 | 1742.13 | 1734.69 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:45:00 | 1611.90 | 1621.72 | 0.00 | ORB-short ORB[1623.80,1636.20] vol=1.6x ATR=5.68 |
| Stop hit — per-position SL triggered | 2026-03-16 11:05:00 | 1617.58 | 1618.40 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 10:05:00 | 1628.00 | 1628.91 | 0.00 | ORB-short ORB[1629.30,1647.80] vol=9.9x ATR=4.95 |
| Stop hit — per-position SL triggered | 2026-03-19 10:10:00 | 1632.95 | 1628.92 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 10:50:00 | 1643.00 | 1637.56 | 0.00 | ORB-long ORB[1621.00,1640.90] vol=3.9x ATR=4.51 |
| Stop hit — per-position SL triggered | 2026-03-20 11:00:00 | 1638.49 | 1638.46 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-03-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:20:00 | 1572.30 | 1577.31 | 0.00 | ORB-short ORB[1576.30,1599.00] vol=2.0x ATR=7.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-24 10:55:00 | 1560.51 | 1572.54 | 0.00 | T1 1.5R @ 1560.51 |
| Target hit | 2026-03-24 11:55:00 | 1569.80 | 1569.49 | 0.00 | Trail-exit close>VWAP |

### Cycle 14 — SELL (started 2026-03-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 11:05:00 | 1548.60 | 1554.78 | 0.00 | ORB-short ORB[1555.00,1574.00] vol=1.6x ATR=4.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 12:20:00 | 1541.71 | 1550.76 | 0.00 | T1 1.5R @ 1541.71 |
| Stop hit — per-position SL triggered | 2026-03-27 12:25:00 | 1548.60 | 1550.41 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:10:00 | 1633.60 | 1641.65 | 0.00 | ORB-short ORB[1642.00,1656.60] vol=5.1x ATR=5.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 11:00:00 | 1625.72 | 1639.67 | 0.00 | T1 1.5R @ 1625.72 |
| Target hit | 2026-04-22 14:45:00 | 1630.00 | 1628.58 | 0.00 | Trail-exit close>VWAP |

### Cycle 16 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:15:00 | 1627.30 | 1621.70 | 0.00 | ORB-long ORB[1614.10,1624.70] vol=2.5x ATR=4.88 |
| Stop hit — per-position SL triggered | 2026-04-29 10:30:00 | 1622.42 | 1622.21 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-05-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:35:00 | 1602.80 | 1595.50 | 0.00 | ORB-long ORB[1585.00,1596.50] vol=1.7x ATR=5.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 09:55:00 | 1610.95 | 1598.35 | 0.00 | T1 1.5R @ 1610.95 |
| Stop hit — per-position SL triggered | 2026-05-04 10:20:00 | 1602.80 | 1599.82 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-05-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:05:00 | 1580.00 | 1589.63 | 0.00 | ORB-short ORB[1586.90,1605.00] vol=1.7x ATR=4.68 |
| Stop hit — per-position SL triggered | 2026-05-05 11:10:00 | 1584.68 | 1589.28 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2026-05-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:40:00 | 1632.90 | 1624.34 | 0.00 | ORB-long ORB[1604.90,1629.00] vol=1.5x ATR=6.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:00:00 | 1642.89 | 1630.98 | 0.00 | T1 1.5R @ 1642.89 |
| Stop hit — per-position SL triggered | 2026-05-06 10:20:00 | 1632.90 | 1632.41 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2026-05-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:40:00 | 1677.80 | 1659.29 | 0.00 | ORB-long ORB[1654.50,1665.40] vol=2.2x ATR=6.06 |
| Stop hit — per-position SL triggered | 2026-05-07 11:05:00 | 1671.74 | 1670.66 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2026-05-08 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:10:00 | 1690.90 | 1698.90 | 0.00 | ORB-short ORB[1705.00,1718.80] vol=2.0x ATR=4.39 |
| Stop hit — per-position SL triggered | 2026-05-08 10:40:00 | 1695.29 | 1695.63 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:45:00 | 1728.30 | 2026-02-09 11:20:00 | 1722.61 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-11 10:50:00 | 1739.90 | 2026-02-11 11:10:00 | 1733.59 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-02-11 10:50:00 | 1739.90 | 2026-02-11 11:20:00 | 1739.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-16 10:45:00 | 1851.60 | 2026-02-16 10:50:00 | 1847.14 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-20 11:10:00 | 1845.80 | 2026-02-20 11:15:00 | 1841.52 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-03-04 10:40:00 | 1736.40 | 2026-03-04 12:15:00 | 1727.17 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-03-04 10:40:00 | 1736.40 | 2026-03-04 13:50:00 | 1736.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-05 11:15:00 | 1738.50 | 2026-03-05 11:50:00 | 1731.32 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-03-05 11:15:00 | 1738.50 | 2026-03-05 14:50:00 | 1738.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-06 10:35:00 | 1717.70 | 2026-03-06 11:50:00 | 1708.12 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-03-06 10:35:00 | 1717.70 | 2026-03-06 14:25:00 | 1709.90 | TARGET_HIT | 0.50 | 0.45% |
| SELL | retest1 | 2026-03-10 10:55:00 | 1720.80 | 2026-03-10 13:35:00 | 1726.60 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-03-11 11:00:00 | 1747.20 | 2026-03-11 11:10:00 | 1742.13 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-16 10:45:00 | 1611.90 | 2026-03-16 11:05:00 | 1617.58 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-03-19 10:05:00 | 1628.00 | 2026-03-19 10:10:00 | 1632.95 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-03-20 10:50:00 | 1643.00 | 2026-03-20 11:00:00 | 1638.49 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-03-24 10:20:00 | 1572.30 | 2026-03-24 10:55:00 | 1560.51 | PARTIAL | 0.50 | 0.75% |
| SELL | retest1 | 2026-03-24 10:20:00 | 1572.30 | 2026-03-24 11:55:00 | 1569.80 | TARGET_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2026-03-27 11:05:00 | 1548.60 | 2026-03-27 12:20:00 | 1541.71 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-03-27 11:05:00 | 1548.60 | 2026-03-27 12:25:00 | 1548.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-22 10:10:00 | 1633.60 | 2026-04-22 11:00:00 | 1625.72 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-04-22 10:10:00 | 1633.60 | 2026-04-22 14:45:00 | 1630.00 | TARGET_HIT | 0.50 | 0.22% |
| BUY | retest1 | 2026-04-29 10:15:00 | 1627.30 | 2026-04-29 10:30:00 | 1622.42 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-04 09:35:00 | 1602.80 | 2026-05-04 09:55:00 | 1610.95 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-05-04 09:35:00 | 1602.80 | 2026-05-04 10:20:00 | 1602.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-05 11:05:00 | 1580.00 | 2026-05-05 11:10:00 | 1584.68 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-06 09:40:00 | 1632.90 | 2026-05-06 10:00:00 | 1642.89 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-05-06 09:40:00 | 1632.90 | 2026-05-06 10:20:00 | 1632.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-07 10:40:00 | 1677.80 | 2026-05-07 11:05:00 | 1671.74 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-05-08 10:10:00 | 1690.90 | 2026-05-08 10:40:00 | 1695.29 | STOP_HIT | 1.00 | -0.26% |
