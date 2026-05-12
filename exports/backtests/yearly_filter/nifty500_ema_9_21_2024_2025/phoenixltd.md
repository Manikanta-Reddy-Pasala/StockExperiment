# Phoenix Mills Ltd. (PHOENIXLTD)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1845.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 149 |
| ALERT1 | 104 |
| ALERT2 | 102 |
| ALERT2_SKIP | 51 |
| ALERT3 | 283 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 8 |
| ENTRY2 | 124 |
| PARTIAL | 14 |
| TARGET_HIT | 12 |
| STOP_HIT | 118 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 144 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 72 / 72
- **Target hits / Stop hits / Partials:** 12 / 118 / 14
- **Avg / median % per leg:** 1.16% / 0.19%
- **Sum % (uncompounded):** 166.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 70 | 42 | 60.0% | 7 | 62 | 1 | 1.50% | 105.0% |
| BUY @ 2nd Alert (retest1) | 9 | 2 | 22.2% | 0 | 8 | 1 | -0.98% | -8.8% |
| BUY @ 3rd Alert (retest2) | 61 | 40 | 65.6% | 7 | 54 | 0 | 1.87% | 113.8% |
| SELL (all) | 74 | 30 | 40.5% | 5 | 56 | 13 | 0.84% | 62.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 74 | 30 | 40.5% | 5 | 56 | 13 | 0.84% | 62.0% |
| retest1 (combined) | 9 | 2 | 22.2% | 0 | 8 | 1 | -0.98% | -8.8% |
| retest2 (combined) | 135 | 70 | 51.9% | 12 | 110 | 13 | 1.30% | 175.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 13:15:00 | 1482.90 | 1452.67 | 1449.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 14:15:00 | 1496.28 | 1461.39 | 1453.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 12:15:00 | 1485.63 | 1486.79 | 1471.46 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 14:15:00 | 1495.73 | 1485.93 | 1472.46 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 09:15:00 | 1518.45 | 1488.69 | 1476.13 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 11:00:00 | 1494.98 | 1492.30 | 1480.10 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 11:45:00 | 1499.70 | 1490.35 | 1480.32 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 12:15:00 | 1466.20 | 1485.52 | 1479.04 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-15 12:15:00 | 1466.20 | 1485.52 | 1479.04 | SL hit (close<ema400) qty=1.00 sl=1479.04 alert=retest1 |

### Cycle 2 — SELL (started 2024-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 09:15:00 | 1521.63 | 1581.06 | 1588.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 09:15:00 | 1513.28 | 1540.29 | 1551.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 11:15:00 | 1551.90 | 1538.28 | 1548.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 11:15:00 | 1551.90 | 1538.28 | 1548.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 11:15:00 | 1551.90 | 1538.28 | 1548.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 11:45:00 | 1553.50 | 1538.28 | 1548.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 12:15:00 | 1556.93 | 1542.01 | 1549.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 12:45:00 | 1557.65 | 1542.01 | 1549.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 14:15:00 | 1524.20 | 1539.89 | 1547.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 14:30:00 | 1557.53 | 1539.89 | 1547.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 1531.00 | 1538.31 | 1545.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 10:15:00 | 1527.28 | 1538.31 | 1545.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 11:15:00 | 1521.30 | 1537.31 | 1544.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-31 12:15:00 | 1586.55 | 1552.01 | 1549.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 12:15:00 | 1586.55 | 1552.01 | 1549.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 1626.28 | 1572.98 | 1560.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 1629.83 | 1638.95 | 1607.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-04 11:00:00 | 1629.83 | 1638.95 | 1607.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 1548.15 | 1620.79 | 1602.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:45:00 | 1542.00 | 1620.79 | 1602.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 1584.25 | 1613.48 | 1600.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 15:00:00 | 1647.03 | 1615.24 | 1603.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-05 09:15:00 | 1500.70 | 1601.29 | 1599.55 | SL hit (close<static) qty=1.00 sl=1536.98 alert=retest2 |

### Cycle 4 — SELL (started 2024-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-05 10:15:00 | 1563.88 | 1593.81 | 1596.30 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 1648.23 | 1604.06 | 1599.20 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-06-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-07 12:15:00 | 1599.13 | 1603.95 | 1604.14 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-06-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 14:15:00 | 1611.25 | 1605.23 | 1604.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 09:15:00 | 1655.48 | 1615.89 | 1609.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 12:15:00 | 1680.15 | 1692.27 | 1674.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-12 12:15:00 | 1680.15 | 1692.27 | 1674.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 12:15:00 | 1680.15 | 1692.27 | 1674.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 13:00:00 | 1680.15 | 1692.27 | 1674.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 14:15:00 | 1704.00 | 1694.79 | 1678.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 14:45:00 | 1680.35 | 1694.79 | 1678.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 1687.25 | 1698.56 | 1683.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 10:00:00 | 1687.25 | 1698.56 | 1683.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 1689.20 | 1696.69 | 1683.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 11:15:00 | 1701.08 | 1696.69 | 1683.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-18 14:15:00 | 1871.19 | 1795.56 | 1756.07 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 13:15:00 | 1776.98 | 1799.72 | 1802.35 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 11:15:00 | 1808.68 | 1803.45 | 1803.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 13:15:00 | 1819.08 | 1808.74 | 1805.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-24 14:15:00 | 1797.90 | 1806.57 | 1805.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 14:15:00 | 1797.90 | 1806.57 | 1805.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 14:15:00 | 1797.90 | 1806.57 | 1805.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 15:00:00 | 1797.90 | 1806.57 | 1805.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 15:15:00 | 1798.00 | 1804.85 | 1804.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-25 09:15:00 | 1830.73 | 1804.85 | 1804.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 12:15:00 | 1794.90 | 1816.43 | 1818.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 12:15:00 | 1794.90 | 1816.43 | 1818.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 13:15:00 | 1787.95 | 1810.73 | 1815.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 11:15:00 | 1795.83 | 1793.13 | 1803.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-28 12:00:00 | 1795.83 | 1793.13 | 1803.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 1791.40 | 1793.07 | 1800.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 15:00:00 | 1791.40 | 1793.07 | 1800.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 1780.00 | 1790.45 | 1798.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:15:00 | 1774.08 | 1790.45 | 1798.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 1777.13 | 1787.79 | 1797.00 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2024-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 09:15:00 | 1820.28 | 1793.28 | 1792.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 10:15:00 | 1832.30 | 1801.08 | 1796.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 09:15:00 | 1856.05 | 1864.33 | 1850.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-08 10:00:00 | 1856.05 | 1864.33 | 1850.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 1928.90 | 1961.68 | 1951.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:45:00 | 1933.33 | 1961.68 | 1951.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 1929.08 | 1955.16 | 1949.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 10:30:00 | 1928.08 | 1955.16 | 1949.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2024-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 12:15:00 | 1922.53 | 1943.48 | 1944.53 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 15:15:00 | 1951.50 | 1945.60 | 1945.18 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 09:15:00 | 1925.75 | 1941.63 | 1943.41 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 11:15:00 | 1969.55 | 1948.87 | 1946.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 12:15:00 | 1982.93 | 1955.69 | 1949.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 12:15:00 | 2022.40 | 2022.49 | 2004.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-18 12:30:00 | 2020.03 | 2022.49 | 2004.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 2006.70 | 2026.73 | 2013.10 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2024-07-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 12:15:00 | 1973.23 | 2001.18 | 2003.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-22 09:15:00 | 1879.90 | 1967.63 | 1986.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 11:15:00 | 1876.38 | 1868.44 | 1908.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-23 12:00:00 | 1876.38 | 1868.44 | 1908.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 1884.98 | 1865.99 | 1896.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 15:00:00 | 1884.98 | 1865.99 | 1896.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 1810.30 | 1855.49 | 1886.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 11:00:00 | 1802.05 | 1844.80 | 1878.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 13:00:00 | 1802.60 | 1831.17 | 1866.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 13:45:00 | 1792.65 | 1824.00 | 1860.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 10:15:00 | 1802.33 | 1773.58 | 1784.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 12:15:00 | 1786.90 | 1781.01 | 1785.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 13:15:00 | 1793.90 | 1781.01 | 1785.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 13:15:00 | 1790.98 | 1783.00 | 1786.43 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-29 14:15:00 | 1815.60 | 1789.52 | 1789.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 14:15:00 | 1815.60 | 1789.52 | 1789.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 09:15:00 | 1896.33 | 1814.40 | 1800.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 13:15:00 | 1833.20 | 1848.88 | 1825.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 13:45:00 | 1830.40 | 1848.88 | 1825.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 1813.00 | 1841.80 | 1827.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 10:00:00 | 1813.00 | 1841.80 | 1827.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 1812.10 | 1835.86 | 1826.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 10:30:00 | 1816.40 | 1835.86 | 1826.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 12:15:00 | 1808.60 | 1828.99 | 1824.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 13:00:00 | 1808.60 | 1828.99 | 1824.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 13:15:00 | 1805.25 | 1824.24 | 1823.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:00:00 | 1805.25 | 1824.24 | 1823.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 14:15:00 | 1801.70 | 1819.73 | 1821.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 09:15:00 | 1762.90 | 1804.39 | 1813.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-01 13:15:00 | 1800.35 | 1793.24 | 1804.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 13:15:00 | 1800.35 | 1793.24 | 1804.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 13:15:00 | 1800.35 | 1793.24 | 1804.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 13:45:00 | 1800.50 | 1793.24 | 1804.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 1807.45 | 1796.08 | 1804.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 15:00:00 | 1807.45 | 1796.08 | 1804.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 15:15:00 | 1801.50 | 1797.17 | 1804.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-02 09:15:00 | 1785.55 | 1797.17 | 1804.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-02 09:45:00 | 1782.05 | 1795.11 | 1802.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-05 09:15:00 | 1606.99 | 1723.03 | 1760.62 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 11:15:00 | 1681.40 | 1658.21 | 1655.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-13 10:15:00 | 1693.50 | 1676.46 | 1670.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-14 09:15:00 | 1673.50 | 1693.98 | 1684.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 09:15:00 | 1673.50 | 1693.98 | 1684.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 1673.50 | 1693.98 | 1684.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-19 09:15:00 | 1726.03 | 1687.56 | 1686.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-19 11:15:00 | 1710.00 | 1696.50 | 1690.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-19 12:00:00 | 1712.10 | 1699.62 | 1692.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-30 14:15:00 | 1898.63 | 1834.56 | 1823.37 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 12:15:00 | 1826.50 | 1834.24 | 1834.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 09:15:00 | 1796.73 | 1822.85 | 1829.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 13:15:00 | 1823.58 | 1813.71 | 1821.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 13:15:00 | 1823.58 | 1813.71 | 1821.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 13:15:00 | 1823.58 | 1813.71 | 1821.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 14:00:00 | 1823.58 | 1813.71 | 1821.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 14:15:00 | 1811.20 | 1813.21 | 1820.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 09:15:00 | 1804.90 | 1813.72 | 1820.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 11:15:00 | 1806.88 | 1810.01 | 1817.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 09:15:00 | 1868.13 | 1815.80 | 1815.85 | SL hit (close>static) qty=1.00 sl=1823.83 alert=retest2 |

### Cycle 21 — BUY (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-06 10:15:00 | 1824.95 | 1817.63 | 1816.68 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-09-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 13:15:00 | 1809.85 | 1816.06 | 1816.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 1797.93 | 1810.65 | 1813.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 1785.45 | 1774.19 | 1789.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 1785.45 | 1774.19 | 1789.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 1785.45 | 1774.19 | 1789.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 10:00:00 | 1760.65 | 1778.99 | 1785.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 15:15:00 | 1672.62 | 1700.14 | 1727.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-16 09:15:00 | 1713.93 | 1682.62 | 1700.59 | SL hit (close>ema200) qty=0.50 sl=1682.62 alert=retest2 |

### Cycle 23 — BUY (started 2024-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 13:15:00 | 1746.95 | 1713.41 | 1710.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 14:15:00 | 1764.68 | 1723.66 | 1715.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 14:15:00 | 1742.58 | 1747.46 | 1735.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-17 15:00:00 | 1742.58 | 1747.46 | 1735.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 1748.33 | 1747.17 | 1737.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:45:00 | 1735.00 | 1747.17 | 1737.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 1745.43 | 1746.82 | 1737.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 10:45:00 | 1741.58 | 1746.82 | 1737.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 11:15:00 | 1748.03 | 1747.06 | 1738.73 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2024-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 15:15:00 | 1720.50 | 1733.11 | 1734.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 12:15:00 | 1697.95 | 1723.55 | 1729.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 1769.48 | 1729.09 | 1730.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 14:15:00 | 1769.48 | 1729.09 | 1730.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 1769.48 | 1729.09 | 1730.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 15:00:00 | 1769.48 | 1729.09 | 1730.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 15:15:00 | 1747.00 | 1732.68 | 1732.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 14:15:00 | 1804.35 | 1755.56 | 1744.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 11:15:00 | 1793.55 | 1797.51 | 1772.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-23 12:00:00 | 1793.55 | 1797.51 | 1772.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 1805.00 | 1801.17 | 1786.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 11:30:00 | 1773.30 | 1801.17 | 1786.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 1797.95 | 1814.41 | 1800.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:45:00 | 1794.85 | 1814.41 | 1800.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 1811.95 | 1813.91 | 1801.52 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 09:15:00 | 1785.00 | 1795.64 | 1796.30 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 10:15:00 | 1816.00 | 1799.71 | 1798.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 15:15:00 | 1823.95 | 1809.53 | 1803.83 | Break + close above crossover candle high |

### Cycle 28 — SELL (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 09:15:00 | 1744.15 | 1796.46 | 1798.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 10:15:00 | 1730.70 | 1783.30 | 1792.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 14:15:00 | 1840.85 | 1775.81 | 1783.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 14:15:00 | 1840.85 | 1775.81 | 1783.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 1840.85 | 1775.81 | 1783.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 15:00:00 | 1840.85 | 1775.81 | 1783.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2024-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 15:15:00 | 1924.00 | 1805.44 | 1796.07 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 11:15:00 | 1771.80 | 1788.16 | 1789.59 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 14:15:00 | 1868.55 | 1800.19 | 1794.36 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 12:15:00 | 1778.70 | 1794.21 | 1794.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 14:15:00 | 1761.00 | 1784.61 | 1789.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 14:15:00 | 1673.55 | 1656.69 | 1690.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 15:00:00 | 1673.55 | 1656.69 | 1690.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 15:15:00 | 1662.20 | 1657.79 | 1687.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-07 09:15:00 | 1689.25 | 1657.79 | 1687.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 1671.65 | 1660.56 | 1686.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 15:15:00 | 1661.00 | 1673.19 | 1683.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 13:15:00 | 1655.00 | 1675.24 | 1680.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 09:15:00 | 1749.65 | 1681.90 | 1680.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 1749.65 | 1681.90 | 1680.76 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 13:15:00 | 1666.95 | 1693.55 | 1695.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 15:15:00 | 1665.00 | 1683.99 | 1690.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 14:15:00 | 1662.85 | 1628.21 | 1635.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 14:15:00 | 1662.85 | 1628.21 | 1635.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 1662.85 | 1628.21 | 1635.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 15:00:00 | 1662.85 | 1628.21 | 1635.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 15:15:00 | 1652.00 | 1632.97 | 1637.32 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 10:15:00 | 1663.00 | 1642.89 | 1641.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 13:15:00 | 1666.75 | 1653.32 | 1646.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 1630.00 | 1654.86 | 1649.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 1630.00 | 1654.86 | 1649.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 1630.00 | 1654.86 | 1649.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 1630.00 | 1654.86 | 1649.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 1628.00 | 1649.49 | 1647.85 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2024-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 11:15:00 | 1626.75 | 1644.94 | 1645.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 13:15:00 | 1617.90 | 1636.26 | 1641.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-22 14:15:00 | 1573.00 | 1551.45 | 1574.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-22 14:15:00 | 1573.00 | 1551.45 | 1574.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 14:15:00 | 1573.00 | 1551.45 | 1574.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-22 15:00:00 | 1573.00 | 1551.45 | 1574.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 15:15:00 | 1580.00 | 1557.16 | 1575.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 09:15:00 | 1524.45 | 1557.16 | 1575.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 15:00:00 | 1565.35 | 1559.79 | 1568.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 09:15:00 | 1557.45 | 1561.43 | 1568.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 1487.08 | 1519.44 | 1541.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 1479.58 | 1519.44 | 1541.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-28 09:15:00 | 1372.01 | 1467.24 | 1503.92 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 12:15:00 | 1522.70 | 1492.62 | 1492.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 1550.90 | 1513.41 | 1503.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 09:15:00 | 1541.75 | 1557.13 | 1536.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 10:00:00 | 1541.75 | 1557.13 | 1536.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 1541.15 | 1553.93 | 1536.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 10:30:00 | 1537.45 | 1553.93 | 1536.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 1532.25 | 1549.60 | 1536.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:00:00 | 1532.25 | 1549.60 | 1536.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 1513.80 | 1542.44 | 1534.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:45:00 | 1514.70 | 1542.44 | 1534.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 1501.05 | 1530.60 | 1531.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 11:15:00 | 1490.60 | 1518.39 | 1525.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 12:15:00 | 1510.70 | 1501.79 | 1510.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 12:15:00 | 1510.70 | 1501.79 | 1510.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 12:15:00 | 1510.70 | 1501.79 | 1510.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 13:00:00 | 1510.70 | 1501.79 | 1510.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 1500.55 | 1501.55 | 1509.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-06 10:30:00 | 1497.85 | 1504.73 | 1508.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 11:15:00 | 1545.50 | 1512.89 | 1511.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 11:15:00 | 1545.50 | 1512.89 | 1511.92 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 1491.00 | 1511.94 | 1514.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 13:15:00 | 1485.80 | 1499.49 | 1507.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 14:15:00 | 1473.90 | 1471.09 | 1485.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-11 15:00:00 | 1473.90 | 1471.09 | 1485.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 1490.30 | 1474.15 | 1484.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:00:00 | 1490.30 | 1474.15 | 1484.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 1494.00 | 1478.12 | 1485.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 11:00:00 | 1494.00 | 1478.12 | 1485.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 11:15:00 | 1508.05 | 1484.11 | 1487.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 12:00:00 | 1508.05 | 1484.11 | 1487.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2024-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 13:15:00 | 1505.35 | 1491.66 | 1490.42 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 15:15:00 | 1468.55 | 1487.31 | 1488.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 1452.45 | 1480.34 | 1485.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 11:15:00 | 1449.05 | 1447.71 | 1461.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 11:30:00 | 1446.60 | 1447.71 | 1461.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 1450.85 | 1431.10 | 1446.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 10:00:00 | 1450.85 | 1431.10 | 1446.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 10:15:00 | 1446.80 | 1434.24 | 1446.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 11:15:00 | 1448.15 | 1434.24 | 1446.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 11:15:00 | 1468.20 | 1441.03 | 1448.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:00:00 | 1468.20 | 1441.03 | 1448.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 12:15:00 | 1468.50 | 1446.53 | 1450.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:30:00 | 1472.00 | 1446.53 | 1450.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 14:15:00 | 1464.25 | 1452.06 | 1452.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 14:45:00 | 1465.55 | 1452.06 | 1452.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 1472.10 | 1456.17 | 1454.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 13:15:00 | 1494.50 | 1470.88 | 1462.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 15:15:00 | 1470.00 | 1473.04 | 1465.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-21 09:15:00 | 1481.35 | 1473.04 | 1465.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 1470.60 | 1472.55 | 1465.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 12:15:00 | 1504.95 | 1477.26 | 1468.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-25 09:15:00 | 1655.45 | 1599.11 | 1550.72 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 10:15:00 | 1785.55 | 1837.68 | 1839.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 12:15:00 | 1775.80 | 1816.19 | 1828.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 09:15:00 | 1819.45 | 1801.18 | 1816.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 09:15:00 | 1819.45 | 1801.18 | 1816.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 1819.45 | 1801.18 | 1816.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:00:00 | 1819.45 | 1801.18 | 1816.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 1822.60 | 1805.46 | 1816.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:30:00 | 1821.30 | 1805.46 | 1816.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 1825.00 | 1809.37 | 1817.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 11:30:00 | 1823.20 | 1809.37 | 1817.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 09:15:00 | 1837.90 | 1821.76 | 1821.10 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 11:15:00 | 1808.75 | 1820.03 | 1820.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 12:15:00 | 1791.15 | 1814.26 | 1817.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 09:15:00 | 1658.00 | 1643.21 | 1679.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 10:00:00 | 1658.00 | 1643.21 | 1679.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 1700.00 | 1661.69 | 1671.83 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2024-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 12:15:00 | 1714.10 | 1681.09 | 1678.94 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 11:15:00 | 1670.10 | 1679.60 | 1680.01 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 14:15:00 | 1734.90 | 1687.80 | 1683.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 10:15:00 | 1749.45 | 1717.68 | 1704.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 14:15:00 | 1667.30 | 1713.03 | 1707.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 14:15:00 | 1667.30 | 1713.03 | 1707.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 1667.30 | 1713.03 | 1707.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 1667.30 | 1713.03 | 1707.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 1683.00 | 1707.02 | 1704.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:15:00 | 1655.85 | 1707.02 | 1704.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2024-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 09:15:00 | 1632.50 | 1692.12 | 1698.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 11:15:00 | 1619.25 | 1667.04 | 1685.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-02 09:15:00 | 1607.85 | 1606.45 | 1630.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-02 09:30:00 | 1614.85 | 1606.45 | 1630.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 1591.60 | 1596.87 | 1612.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-03 10:30:00 | 1586.65 | 1595.00 | 1610.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-08 13:15:00 | 1647.00 | 1584.88 | 1577.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 13:15:00 | 1647.00 | 1584.88 | 1577.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-08 14:15:00 | 1750.50 | 1618.01 | 1593.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 13:15:00 | 1647.55 | 1662.85 | 1631.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-09 14:00:00 | 1647.55 | 1662.85 | 1631.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 14:15:00 | 1630.00 | 1656.28 | 1631.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 15:00:00 | 1630.00 | 1656.28 | 1631.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 15:15:00 | 1647.00 | 1654.42 | 1632.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 09:15:00 | 1661.25 | 1654.42 | 1632.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 13:30:00 | 1654.80 | 1656.67 | 1642.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 14:00:00 | 1654.05 | 1656.67 | 1642.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-13 09:15:00 | 1587.65 | 1643.13 | 1640.05 | SL hit (close<static) qty=1.00 sl=1625.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 10:15:00 | 1593.60 | 1633.22 | 1635.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 11:15:00 | 1569.55 | 1620.49 | 1629.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 1570.00 | 1563.37 | 1593.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 1570.00 | 1563.37 | 1593.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 1570.00 | 1563.37 | 1593.03 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2025-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 11:15:00 | 1631.00 | 1595.98 | 1593.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 12:15:00 | 1657.60 | 1608.31 | 1599.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-16 14:15:00 | 1654.95 | 1656.40 | 1636.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-16 15:00:00 | 1654.95 | 1656.40 | 1636.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 1653.60 | 1656.41 | 1639.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 12:45:00 | 1667.60 | 1658.98 | 1645.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 09:15:00 | 1627.00 | 1646.94 | 1643.57 | SL hit (close<static) qty=1.00 sl=1638.15 alert=retest2 |

### Cycle 54 — SELL (started 2025-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 11:15:00 | 1618.35 | 1638.37 | 1640.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-20 14:15:00 | 1609.60 | 1626.20 | 1633.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 1515.70 | 1488.32 | 1526.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 1515.70 | 1488.32 | 1526.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1515.70 | 1488.32 | 1526.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 1515.70 | 1488.32 | 1526.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1517.90 | 1494.24 | 1525.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 12:15:00 | 1507.05 | 1498.39 | 1524.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 13:15:00 | 1514.50 | 1501.73 | 1523.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 14:15:00 | 1530.15 | 1511.01 | 1524.55 | SL hit (close>static) qty=1.00 sl=1528.60 alert=retest2 |

### Cycle 55 — BUY (started 2025-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 15:15:00 | 1517.45 | 1505.53 | 1505.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 09:15:00 | 1541.05 | 1512.63 | 1508.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-04 10:15:00 | 1766.50 | 1783.38 | 1736.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-04 10:45:00 | 1771.20 | 1783.38 | 1736.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 1740.05 | 1774.70 | 1753.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:00:00 | 1740.05 | 1774.70 | 1753.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 1737.65 | 1767.29 | 1751.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 11:00:00 | 1737.65 | 1767.29 | 1751.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2025-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 13:15:00 | 1706.00 | 1741.46 | 1742.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-05 14:15:00 | 1701.25 | 1733.42 | 1738.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 10:15:00 | 1658.05 | 1648.75 | 1678.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-07 11:00:00 | 1658.05 | 1648.75 | 1678.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 1583.10 | 1573.11 | 1592.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:45:00 | 1592.00 | 1573.11 | 1592.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 1605.95 | 1579.68 | 1593.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 1605.95 | 1579.68 | 1593.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 1591.90 | 1582.12 | 1593.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:30:00 | 1604.15 | 1582.12 | 1593.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 14:15:00 | 1597.15 | 1585.13 | 1593.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 14:45:00 | 1599.55 | 1585.13 | 1593.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 1609.95 | 1590.09 | 1595.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 09:15:00 | 1591.45 | 1590.09 | 1595.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-13 12:15:00 | 1608.50 | 1597.98 | 1597.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2025-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 12:15:00 | 1608.50 | 1597.98 | 1597.89 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 15:15:00 | 1592.85 | 1597.89 | 1597.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 09:15:00 | 1565.10 | 1591.33 | 1594.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-14 14:15:00 | 1570.15 | 1567.67 | 1579.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-14 15:00:00 | 1570.15 | 1567.67 | 1579.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 15:15:00 | 1555.10 | 1565.16 | 1577.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 09:15:00 | 1550.00 | 1565.16 | 1577.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-18 13:15:00 | 1586.20 | 1564.31 | 1563.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 13:15:00 | 1586.20 | 1564.31 | 1563.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 14:15:00 | 1597.30 | 1570.91 | 1566.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-19 11:15:00 | 1582.10 | 1584.48 | 1575.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-19 12:00:00 | 1582.10 | 1584.48 | 1575.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 12:15:00 | 1574.80 | 1582.55 | 1575.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-19 12:45:00 | 1572.20 | 1582.55 | 1575.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 13:15:00 | 1559.15 | 1577.87 | 1573.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-19 14:00:00 | 1559.15 | 1577.87 | 1573.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 14:15:00 | 1556.15 | 1573.52 | 1572.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-19 15:15:00 | 1559.15 | 1573.52 | 1572.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2025-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-19 15:15:00 | 1559.15 | 1570.65 | 1571.04 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 13:15:00 | 1587.15 | 1572.23 | 1571.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 14:15:00 | 1599.35 | 1577.65 | 1573.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 1567.60 | 1578.90 | 1575.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 1567.60 | 1578.90 | 1575.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 1567.60 | 1578.90 | 1575.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 1567.60 | 1578.90 | 1575.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 1574.25 | 1577.97 | 1575.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 12:15:00 | 1584.15 | 1578.70 | 1575.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 14:15:00 | 1589.45 | 1580.16 | 1576.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 15:00:00 | 1590.35 | 1582.20 | 1578.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 09:15:00 | 1562.75 | 1577.16 | 1576.49 | SL hit (close<static) qty=1.00 sl=1567.40 alert=retest2 |

### Cycle 62 — SELL (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 10:15:00 | 1570.00 | 1575.72 | 1575.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 13:15:00 | 1559.20 | 1570.12 | 1573.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 14:15:00 | 1573.85 | 1570.86 | 1573.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 14:15:00 | 1573.85 | 1570.86 | 1573.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 14:15:00 | 1573.85 | 1570.86 | 1573.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 14:45:00 | 1577.70 | 1570.86 | 1573.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 15:15:00 | 1572.85 | 1571.26 | 1573.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 09:15:00 | 1530.60 | 1571.26 | 1573.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-25 12:15:00 | 1580.00 | 1568.26 | 1570.48 | SL hit (close>static) qty=1.00 sl=1577.35 alert=retest2 |

### Cycle 63 — BUY (started 2025-03-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 13:15:00 | 1545.25 | 1528.76 | 1528.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-03 14:15:00 | 1552.00 | 1533.41 | 1530.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-04 11:15:00 | 1530.35 | 1536.11 | 1533.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 11:15:00 | 1530.35 | 1536.11 | 1533.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 11:15:00 | 1530.35 | 1536.11 | 1533.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:45:00 | 1530.95 | 1536.11 | 1533.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 12:15:00 | 1535.30 | 1535.95 | 1533.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-04 13:00:00 | 1535.30 | 1535.95 | 1533.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 13:15:00 | 1551.15 | 1538.99 | 1535.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-04 14:45:00 | 1558.90 | 1541.14 | 1536.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-04 15:15:00 | 1558.00 | 1541.14 | 1536.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-05 11:30:00 | 1567.20 | 1551.99 | 1543.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 15:15:00 | 1551.00 | 1573.32 | 1574.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 1551.00 | 1573.32 | 1574.57 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 09:15:00 | 1589.45 | 1576.54 | 1575.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 10:15:00 | 1623.25 | 1585.89 | 1580.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 10:15:00 | 1614.80 | 1623.89 | 1606.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-12 11:00:00 | 1614.80 | 1623.89 | 1606.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 1604.95 | 1620.10 | 1606.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 12:00:00 | 1604.95 | 1620.10 | 1606.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 1605.75 | 1617.23 | 1606.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 12:45:00 | 1607.00 | 1617.23 | 1606.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 13:15:00 | 1607.85 | 1615.35 | 1606.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 14:00:00 | 1607.85 | 1615.35 | 1606.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 1600.00 | 1612.28 | 1605.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 15:00:00 | 1600.00 | 1612.28 | 1605.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 1600.95 | 1610.02 | 1605.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:15:00 | 1590.10 | 1610.02 | 1605.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 10:15:00 | 1575.10 | 1598.96 | 1601.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 11:15:00 | 1564.00 | 1591.97 | 1597.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 13:15:00 | 1574.85 | 1566.04 | 1576.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-17 14:00:00 | 1574.85 | 1566.04 | 1576.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 1557.45 | 1564.32 | 1574.71 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 11:15:00 | 1606.35 | 1580.54 | 1579.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 15:15:00 | 1610.05 | 1596.39 | 1588.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 09:15:00 | 1550.80 | 1587.27 | 1584.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-19 09:15:00 | 1550.80 | 1587.27 | 1584.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 1550.80 | 1587.27 | 1584.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 10:00:00 | 1550.80 | 1587.27 | 1584.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-19 10:15:00 | 1554.00 | 1580.61 | 1582.03 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 12:15:00 | 1619.40 | 1581.61 | 1577.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-21 09:15:00 | 1682.75 | 1618.95 | 1597.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 10:15:00 | 1662.00 | 1675.61 | 1658.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-25 10:15:00 | 1662.00 | 1675.61 | 1658.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 1662.00 | 1675.61 | 1658.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:45:00 | 1665.15 | 1675.61 | 1658.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 1660.05 | 1672.50 | 1658.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 1656.40 | 1672.50 | 1658.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 1655.00 | 1669.00 | 1658.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:00:00 | 1655.00 | 1669.00 | 1658.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 1665.45 | 1668.29 | 1659.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 1678.40 | 1663.93 | 1658.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 10:15:00 | 1647.20 | 1658.51 | 1656.95 | SL hit (close<static) qty=1.00 sl=1647.95 alert=retest2 |

### Cycle 70 — SELL (started 2025-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 12:15:00 | 1642.50 | 1653.38 | 1654.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 14:15:00 | 1641.20 | 1649.26 | 1652.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 13:15:00 | 1649.00 | 1639.95 | 1645.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 13:15:00 | 1649.00 | 1639.95 | 1645.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 1649.00 | 1639.95 | 1645.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:00:00 | 1649.00 | 1639.95 | 1645.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 1663.10 | 1644.58 | 1646.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 1663.10 | 1644.58 | 1646.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 1661.00 | 1647.86 | 1648.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 09:15:00 | 1644.95 | 1647.86 | 1648.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 09:15:00 | 1662.20 | 1650.73 | 1649.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 1662.20 | 1650.73 | 1649.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 11:15:00 | 1674.00 | 1657.67 | 1652.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 14:15:00 | 1643.00 | 1657.09 | 1654.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 14:15:00 | 1643.00 | 1657.09 | 1654.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 1643.00 | 1657.09 | 1654.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 15:00:00 | 1643.00 | 1657.09 | 1654.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 1635.75 | 1652.82 | 1652.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:15:00 | 1640.00 | 1652.82 | 1652.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 09:15:00 | 1628.40 | 1647.94 | 1650.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 10:15:00 | 1599.25 | 1638.20 | 1645.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 09:15:00 | 1647.00 | 1620.70 | 1630.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 1647.00 | 1620.70 | 1630.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 1647.00 | 1620.70 | 1630.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 09:30:00 | 1650.40 | 1620.70 | 1630.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 1650.60 | 1626.68 | 1632.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 11:00:00 | 1650.60 | 1626.68 | 1632.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2025-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 13:15:00 | 1650.10 | 1636.91 | 1636.30 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 10:15:00 | 1627.45 | 1635.91 | 1636.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-03 12:15:00 | 1612.95 | 1629.95 | 1633.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-03 14:15:00 | 1631.20 | 1629.61 | 1632.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 14:15:00 | 1631.20 | 1629.61 | 1632.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 14:15:00 | 1631.20 | 1629.61 | 1632.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 15:00:00 | 1631.20 | 1629.61 | 1632.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 15:15:00 | 1624.35 | 1628.56 | 1631.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-04 09:15:00 | 1615.55 | 1628.56 | 1631.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1599.10 | 1622.67 | 1628.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 10:15:00 | 1591.40 | 1622.67 | 1628.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 12:15:00 | 1594.15 | 1614.96 | 1624.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 12:45:00 | 1592.05 | 1608.99 | 1620.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 1511.83 | 1583.43 | 1603.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 1514.44 | 1583.43 | 1603.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 1512.45 | 1583.43 | 1603.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-08 09:15:00 | 1569.00 | 1562.65 | 1581.02 | SL hit (close>ema200) qty=0.50 sl=1562.65 alert=retest2 |

### Cycle 75 — BUY (started 2025-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 14:15:00 | 1547.70 | 1528.02 | 1525.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 15:15:00 | 1554.00 | 1533.22 | 1528.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-21 13:15:00 | 1608.70 | 1611.76 | 1594.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-21 13:45:00 | 1608.50 | 1611.76 | 1594.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 1662.30 | 1666.84 | 1660.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 1662.30 | 1666.84 | 1660.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 1651.60 | 1663.79 | 1660.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:30:00 | 1650.30 | 1663.79 | 1660.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 11:15:00 | 1648.50 | 1660.73 | 1659.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:45:00 | 1640.10 | 1660.73 | 1659.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2025-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 12:15:00 | 1645.80 | 1657.75 | 1657.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 13:15:00 | 1631.90 | 1652.58 | 1655.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 13:15:00 | 1634.40 | 1630.86 | 1640.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 14:00:00 | 1634.40 | 1630.86 | 1640.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 1652.90 | 1635.41 | 1640.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 10:00:00 | 1652.90 | 1635.41 | 1640.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 1651.10 | 1638.55 | 1641.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 11:15:00 | 1657.10 | 1638.55 | 1641.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 1645.20 | 1639.88 | 1641.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 13:15:00 | 1632.40 | 1640.06 | 1641.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 14:45:00 | 1636.60 | 1637.52 | 1640.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-30 09:15:00 | 1672.70 | 1644.49 | 1642.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 09:15:00 | 1672.70 | 1644.49 | 1642.85 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 09:15:00 | 1568.00 | 1641.39 | 1645.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 11:15:00 | 1539.70 | 1606.96 | 1628.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 14:15:00 | 1555.00 | 1542.19 | 1569.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-05 15:00:00 | 1555.00 | 1542.19 | 1569.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 1532.80 | 1510.02 | 1525.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 1532.80 | 1510.02 | 1525.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 1526.10 | 1513.24 | 1525.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:30:00 | 1498.90 | 1515.83 | 1522.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 09:30:00 | 1510.00 | 1485.65 | 1495.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 10:45:00 | 1513.10 | 1491.36 | 1497.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 13:15:00 | 1531.30 | 1504.76 | 1502.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 1531.30 | 1504.76 | 1502.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 1537.00 | 1516.80 | 1508.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 15:15:00 | 1533.10 | 1538.03 | 1530.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 09:15:00 | 1527.80 | 1538.03 | 1530.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 1532.10 | 1536.84 | 1530.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:30:00 | 1515.40 | 1536.84 | 1530.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 1530.40 | 1535.55 | 1530.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:00:00 | 1530.40 | 1535.55 | 1530.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 1537.60 | 1535.96 | 1531.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:00:00 | 1546.30 | 1538.03 | 1532.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 11:15:00 | 1549.80 | 1544.96 | 1538.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 15:15:00 | 1590.00 | 1597.88 | 1598.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 15:15:00 | 1590.00 | 1597.88 | 1598.76 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 1607.40 | 1599.78 | 1599.54 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 10:15:00 | 1587.30 | 1597.29 | 1598.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 11:15:00 | 1573.10 | 1592.45 | 1596.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 12:15:00 | 1593.00 | 1592.56 | 1595.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-26 13:00:00 | 1593.00 | 1592.56 | 1595.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 1600.00 | 1594.05 | 1596.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:00:00 | 1600.00 | 1594.05 | 1596.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 1598.50 | 1594.94 | 1596.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:30:00 | 1600.00 | 1594.94 | 1596.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 1598.00 | 1595.55 | 1596.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 1607.70 | 1595.55 | 1596.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1599.50 | 1596.34 | 1596.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:30:00 | 1601.00 | 1596.34 | 1596.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 10:15:00 | 1607.30 | 1598.53 | 1597.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 14:15:00 | 1615.40 | 1604.09 | 1600.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 09:15:00 | 1593.10 | 1602.45 | 1600.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 09:15:00 | 1593.10 | 1602.45 | 1600.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 1593.10 | 1602.45 | 1600.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:30:00 | 1590.20 | 1602.45 | 1600.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 1599.10 | 1601.78 | 1600.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 10:30:00 | 1596.90 | 1601.78 | 1600.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 1599.70 | 1601.37 | 1600.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 12:30:00 | 1604.50 | 1600.95 | 1600.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 13:15:00 | 1596.00 | 1599.96 | 1599.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 13:15:00 | 1596.00 | 1599.96 | 1599.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 14:15:00 | 1591.60 | 1598.29 | 1599.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 10:15:00 | 1567.10 | 1555.71 | 1567.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 10:15:00 | 1567.10 | 1555.71 | 1567.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 1567.10 | 1555.71 | 1567.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:00:00 | 1567.10 | 1555.71 | 1567.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 1582.60 | 1561.09 | 1568.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 12:00:00 | 1582.60 | 1561.09 | 1568.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 1578.30 | 1564.53 | 1569.35 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2025-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 15:15:00 | 1589.60 | 1575.42 | 1573.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 12:15:00 | 1596.90 | 1579.44 | 1576.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 09:15:00 | 1583.40 | 1593.08 | 1584.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 1583.40 | 1593.08 | 1584.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 1583.40 | 1593.08 | 1584.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:30:00 | 1584.00 | 1593.08 | 1584.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 1581.20 | 1590.71 | 1584.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:00:00 | 1581.20 | 1590.71 | 1584.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 1578.00 | 1588.17 | 1583.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:30:00 | 1575.50 | 1588.17 | 1583.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 1578.00 | 1584.57 | 1582.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:30:00 | 1573.90 | 1584.57 | 1582.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 1579.60 | 1584.90 | 1583.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:45:00 | 1581.30 | 1584.90 | 1583.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 1591.50 | 1586.22 | 1584.30 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 1569.60 | 1583.00 | 1583.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 14:15:00 | 1562.20 | 1578.84 | 1581.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 09:15:00 | 1576.50 | 1576.17 | 1579.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 1576.50 | 1576.17 | 1579.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 1576.50 | 1576.17 | 1579.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:30:00 | 1572.30 | 1576.17 | 1579.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 1575.00 | 1575.94 | 1579.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:45:00 | 1577.80 | 1575.94 | 1579.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 1589.10 | 1578.57 | 1580.10 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2025-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 13:15:00 | 1590.80 | 1582.86 | 1581.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 14:15:00 | 1595.90 | 1585.47 | 1583.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 14:15:00 | 1626.50 | 1631.74 | 1618.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 14:45:00 | 1622.50 | 1631.74 | 1618.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 1644.70 | 1648.06 | 1633.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 1640.80 | 1648.06 | 1633.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 1627.80 | 1644.32 | 1635.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:45:00 | 1627.80 | 1644.32 | 1635.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 1619.00 | 1639.25 | 1634.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 1619.00 | 1639.25 | 1634.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 1614.30 | 1629.34 | 1630.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 15:15:00 | 1600.00 | 1616.15 | 1623.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 1599.80 | 1598.92 | 1609.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 1599.80 | 1598.92 | 1609.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 1599.60 | 1596.28 | 1604.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:30:00 | 1602.20 | 1596.28 | 1604.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 1613.10 | 1600.00 | 1604.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 1613.10 | 1600.00 | 1604.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1615.70 | 1603.14 | 1605.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:45:00 | 1619.80 | 1603.14 | 1605.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 09:15:00 | 1626.80 | 1609.96 | 1608.65 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 12:15:00 | 1598.00 | 1612.57 | 1613.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 1580.20 | 1602.70 | 1608.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 1587.10 | 1576.22 | 1588.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 1587.10 | 1576.22 | 1588.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1587.10 | 1576.22 | 1588.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:30:00 | 1587.80 | 1576.22 | 1588.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1605.00 | 1581.98 | 1590.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 1605.00 | 1581.98 | 1590.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1606.20 | 1586.82 | 1591.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:45:00 | 1605.90 | 1586.82 | 1591.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 1617.70 | 1597.74 | 1595.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 09:15:00 | 1623.80 | 1605.52 | 1599.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 11:15:00 | 1615.60 | 1618.79 | 1612.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 12:15:00 | 1608.40 | 1616.71 | 1611.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 12:15:00 | 1608.40 | 1616.71 | 1611.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:00:00 | 1608.40 | 1616.71 | 1611.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 1590.20 | 1611.41 | 1609.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 14:00:00 | 1590.20 | 1611.41 | 1609.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 15:15:00 | 1600.00 | 1607.75 | 1608.49 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 14:15:00 | 1621.50 | 1610.36 | 1608.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 14:15:00 | 1629.90 | 1618.28 | 1613.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 15:15:00 | 1618.00 | 1618.23 | 1614.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 09:15:00 | 1626.70 | 1618.23 | 1614.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1610.90 | 1616.76 | 1613.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:00:00 | 1610.90 | 1616.76 | 1613.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 1605.40 | 1614.49 | 1613.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 11:00:00 | 1605.40 | 1614.49 | 1613.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 11:15:00 | 1592.90 | 1610.17 | 1611.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 12:15:00 | 1582.10 | 1604.56 | 1608.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 10:15:00 | 1587.20 | 1586.93 | 1596.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-30 11:00:00 | 1587.20 | 1586.93 | 1596.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 1537.50 | 1517.16 | 1525.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 1537.90 | 1517.16 | 1525.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 1540.90 | 1521.91 | 1526.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:30:00 | 1548.30 | 1521.91 | 1526.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 1531.10 | 1525.03 | 1527.25 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2025-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 14:15:00 | 1542.70 | 1530.40 | 1529.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 09:15:00 | 1546.80 | 1535.36 | 1531.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 15:15:00 | 1543.40 | 1545.46 | 1539.45 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-08 09:15:00 | 1559.90 | 1545.46 | 1539.45 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1517.40 | 1554.93 | 1550.40 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 1517.40 | 1554.93 | 1550.40 | SL hit (close<ema400) qty=1.00 sl=1550.40 alert=retest1 |

### Cycle 96 — SELL (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 11:15:00 | 1515.50 | 1541.97 | 1544.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 10:15:00 | 1504.00 | 1521.78 | 1532.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 1504.40 | 1497.55 | 1507.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 09:30:00 | 1504.80 | 1497.55 | 1507.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 1502.10 | 1498.46 | 1506.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:30:00 | 1501.30 | 1500.43 | 1506.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 13:00:00 | 1501.50 | 1500.64 | 1506.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 14:15:00 | 1512.20 | 1503.65 | 1506.72 | SL hit (close>static) qty=1.00 sl=1511.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 11:15:00 | 1503.50 | 1495.83 | 1495.51 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 1489.60 | 1496.31 | 1496.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 12:15:00 | 1486.10 | 1493.15 | 1494.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 1491.10 | 1490.05 | 1492.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 1491.10 | 1490.05 | 1492.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1491.10 | 1490.05 | 1492.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:30:00 | 1492.50 | 1490.05 | 1492.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 1487.00 | 1489.44 | 1492.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:30:00 | 1490.00 | 1489.44 | 1492.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1492.70 | 1490.09 | 1492.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:30:00 | 1492.70 | 1490.09 | 1492.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 1487.20 | 1489.51 | 1491.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:30:00 | 1491.50 | 1489.51 | 1491.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 1492.40 | 1490.09 | 1491.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:30:00 | 1491.40 | 1490.09 | 1491.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 1492.40 | 1490.55 | 1491.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 14:30:00 | 1492.40 | 1490.55 | 1491.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 1495.90 | 1491.62 | 1492.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 1481.80 | 1491.62 | 1492.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1470.60 | 1487.42 | 1490.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:30:00 | 1465.40 | 1478.19 | 1484.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 09:30:00 | 1461.00 | 1472.09 | 1479.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 1522.60 | 1464.64 | 1464.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-25 09:15:00 | 1522.60 | 1464.64 | 1464.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-25 11:15:00 | 1537.90 | 1489.43 | 1476.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 09:15:00 | 1506.60 | 1511.49 | 1494.63 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 10:45:00 | 1524.60 | 1513.39 | 1497.02 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 11:15:00 | 1524.10 | 1513.39 | 1497.02 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 1492.60 | 1508.94 | 1502.44 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-29 09:15:00 | 1492.60 | 1508.94 | 1502.44 | SL hit (close<ema400) qty=1.00 sl=1502.44 alert=retest1 |

### Cycle 100 — SELL (started 2025-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 15:15:00 | 1499.00 | 1502.79 | 1502.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 1477.20 | 1497.67 | 1500.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 13:15:00 | 1465.90 | 1462.37 | 1471.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 14:00:00 | 1465.90 | 1462.37 | 1471.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1473.90 | 1464.68 | 1471.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:30:00 | 1480.00 | 1464.68 | 1471.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 1473.10 | 1466.36 | 1471.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 1475.90 | 1466.36 | 1471.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 1471.90 | 1467.20 | 1470.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 11:00:00 | 1471.90 | 1467.20 | 1470.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 1475.60 | 1468.88 | 1471.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:00:00 | 1475.60 | 1468.88 | 1471.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 1470.70 | 1469.25 | 1471.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:30:00 | 1475.00 | 1469.25 | 1471.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 1472.50 | 1469.90 | 1471.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:45:00 | 1474.20 | 1469.90 | 1471.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 1479.00 | 1471.72 | 1472.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 15:00:00 | 1479.00 | 1471.72 | 1472.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 1470.20 | 1471.41 | 1471.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:15:00 | 1456.00 | 1471.41 | 1471.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 09:15:00 | 1453.30 | 1439.18 | 1438.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 1453.30 | 1439.18 | 1438.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 10:15:00 | 1461.30 | 1443.60 | 1440.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 14:15:00 | 1435.90 | 1445.45 | 1442.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 14:15:00 | 1435.90 | 1445.45 | 1442.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 1435.90 | 1445.45 | 1442.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 1435.90 | 1445.45 | 1442.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 1432.00 | 1442.76 | 1441.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 1434.70 | 1441.15 | 1441.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 1422.00 | 1437.32 | 1439.27 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 1481.00 | 1442.21 | 1439.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 12:15:00 | 1491.00 | 1463.60 | 1451.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 09:15:00 | 1564.00 | 1567.68 | 1545.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 10:00:00 | 1564.00 | 1567.68 | 1545.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1574.60 | 1583.43 | 1573.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 1569.50 | 1583.43 | 1573.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 1570.40 | 1580.82 | 1573.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:00:00 | 1570.40 | 1580.82 | 1573.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 1571.40 | 1578.94 | 1573.42 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2025-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 15:15:00 | 1557.00 | 1568.60 | 1569.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 1539.50 | 1562.78 | 1567.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 1507.40 | 1507.10 | 1521.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 09:45:00 | 1512.10 | 1507.10 | 1521.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1518.00 | 1509.28 | 1521.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 1518.00 | 1509.28 | 1521.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 1517.10 | 1512.33 | 1520.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:30:00 | 1519.00 | 1512.33 | 1520.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 1516.90 | 1513.77 | 1520.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:30:00 | 1519.40 | 1513.77 | 1520.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 1517.30 | 1514.48 | 1519.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 1570.10 | 1514.48 | 1519.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 1566.90 | 1524.96 | 1524.16 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 13:15:00 | 1513.80 | 1533.17 | 1535.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 11:15:00 | 1495.50 | 1515.59 | 1525.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 1513.90 | 1502.95 | 1510.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 1513.90 | 1502.95 | 1510.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 1513.90 | 1502.95 | 1510.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 1513.90 | 1502.95 | 1510.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 1518.60 | 1506.08 | 1511.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:45:00 | 1519.60 | 1506.08 | 1511.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 10:15:00 | 1527.80 | 1515.95 | 1515.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 11:15:00 | 1535.60 | 1519.88 | 1517.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 11:15:00 | 1525.80 | 1529.05 | 1523.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 11:15:00 | 1525.80 | 1529.05 | 1523.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 1525.80 | 1529.05 | 1523.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:00:00 | 1525.80 | 1529.05 | 1523.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 1552.90 | 1533.82 | 1526.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:30:00 | 1526.70 | 1533.82 | 1526.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 1555.90 | 1556.10 | 1549.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:00:00 | 1555.90 | 1556.10 | 1549.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 1551.50 | 1555.56 | 1551.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:00:00 | 1551.50 | 1555.56 | 1551.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 1556.30 | 1555.71 | 1551.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:30:00 | 1554.10 | 1555.71 | 1551.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1573.70 | 1560.78 | 1555.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 13:15:00 | 1596.70 | 1571.22 | 1562.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 1598.40 | 1580.06 | 1568.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 14:15:00 | 1626.70 | 1628.64 | 1628.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 14:15:00 | 1626.70 | 1628.64 | 1628.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 15:15:00 | 1621.70 | 1627.25 | 1628.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 11:15:00 | 1568.30 | 1567.13 | 1582.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-26 11:45:00 | 1567.20 | 1567.13 | 1582.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1572.40 | 1559.64 | 1571.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 1541.90 | 1557.14 | 1565.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 12:15:00 | 1572.00 | 1561.66 | 1561.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 1572.00 | 1561.66 | 1561.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 1576.20 | 1564.57 | 1562.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 14:15:00 | 1562.20 | 1564.10 | 1562.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 14:15:00 | 1562.20 | 1564.10 | 1562.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 1562.20 | 1564.10 | 1562.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:00:00 | 1562.20 | 1564.10 | 1562.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 1559.60 | 1563.20 | 1562.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 1545.50 | 1563.20 | 1562.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 09:15:00 | 1551.70 | 1560.90 | 1561.60 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 11:15:00 | 1566.40 | 1560.72 | 1559.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 13:15:00 | 1585.80 | 1567.54 | 1563.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 10:15:00 | 1599.90 | 1600.93 | 1589.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 14:15:00 | 1589.70 | 1597.05 | 1590.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 1589.70 | 1597.05 | 1590.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 1589.70 | 1597.05 | 1590.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 1591.40 | 1595.92 | 1590.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 1588.70 | 1595.92 | 1590.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1582.80 | 1593.30 | 1590.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 1576.20 | 1593.30 | 1590.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 1589.20 | 1592.48 | 1590.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 12:30:00 | 1592.50 | 1592.15 | 1590.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 15:00:00 | 1595.20 | 1593.78 | 1591.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 09:30:00 | 1603.60 | 1595.51 | 1592.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 09:45:00 | 1593.90 | 1599.61 | 1597.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 1594.90 | 1598.67 | 1597.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:00:00 | 1618.90 | 1602.71 | 1599.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 14:15:00 | 1657.80 | 1673.48 | 1674.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 14:15:00 | 1657.80 | 1673.48 | 1674.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-21 13:15:00 | 1651.70 | 1666.16 | 1670.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 09:15:00 | 1691.00 | 1668.53 | 1670.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 1691.00 | 1668.53 | 1670.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1691.00 | 1668.53 | 1670.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 1691.00 | 1668.53 | 1670.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 1686.90 | 1672.20 | 1672.39 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 11:15:00 | 1699.90 | 1677.74 | 1674.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 12:15:00 | 1709.90 | 1696.58 | 1690.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 1702.00 | 1702.14 | 1695.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 10:00:00 | 1702.00 | 1702.14 | 1695.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 1697.40 | 1702.18 | 1696.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:45:00 | 1699.60 | 1702.18 | 1696.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 1694.00 | 1700.54 | 1696.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:00:00 | 1694.00 | 1700.54 | 1696.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 1699.90 | 1700.41 | 1696.93 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 13:15:00 | 1691.10 | 1695.84 | 1696.20 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 14:15:00 | 1705.90 | 1697.85 | 1697.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 15:15:00 | 1714.00 | 1701.08 | 1698.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 12:15:00 | 1699.30 | 1703.42 | 1700.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 12:15:00 | 1699.30 | 1703.42 | 1700.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 1699.30 | 1703.42 | 1700.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:00:00 | 1699.30 | 1703.42 | 1700.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 1697.00 | 1702.13 | 1700.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:30:00 | 1698.90 | 1702.13 | 1700.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 1703.00 | 1702.31 | 1700.63 | EMA400 retest candle locked (from upside) |

### Cycle 116 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 1688.20 | 1699.84 | 1699.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 12:15:00 | 1681.20 | 1694.30 | 1697.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 15:15:00 | 1690.10 | 1690.04 | 1694.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 15:15:00 | 1690.10 | 1690.04 | 1694.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 1690.10 | 1690.04 | 1694.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 1698.80 | 1690.04 | 1694.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1719.50 | 1695.93 | 1696.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 1719.50 | 1695.93 | 1696.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 10:15:00 | 1724.80 | 1701.71 | 1699.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 12:15:00 | 1751.50 | 1716.19 | 1706.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 1747.60 | 1753.33 | 1738.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 10:00:00 | 1747.60 | 1753.33 | 1738.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 1745.20 | 1752.16 | 1743.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:00:00 | 1745.20 | 1752.16 | 1743.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 1738.90 | 1749.51 | 1743.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:15:00 | 1744.60 | 1749.51 | 1743.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 1744.90 | 1748.59 | 1743.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:30:00 | 1740.60 | 1748.59 | 1743.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 1768.20 | 1752.51 | 1745.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 11:45:00 | 1770.10 | 1756.61 | 1748.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 14:15:00 | 1773.80 | 1762.62 | 1752.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 10:15:00 | 1740.90 | 1753.30 | 1754.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 1740.90 | 1753.30 | 1754.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 11:15:00 | 1733.10 | 1749.26 | 1752.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 11:15:00 | 1740.10 | 1740.01 | 1745.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-12 12:00:00 | 1740.10 | 1740.01 | 1745.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1736.90 | 1735.99 | 1740.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:30:00 | 1743.30 | 1735.99 | 1740.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 1735.00 | 1735.79 | 1740.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:00:00 | 1735.00 | 1735.79 | 1740.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 1733.50 | 1724.07 | 1730.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:00:00 | 1733.50 | 1724.07 | 1730.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 1734.00 | 1726.05 | 1730.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 14:45:00 | 1729.00 | 1730.79 | 1732.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 1727.80 | 1731.85 | 1732.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 10:15:00 | 1742.70 | 1733.10 | 1732.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 1742.70 | 1733.10 | 1732.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 13:15:00 | 1750.00 | 1739.67 | 1736.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 1718.80 | 1736.94 | 1735.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 1718.80 | 1736.94 | 1735.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1718.80 | 1736.94 | 1735.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 1718.80 | 1736.94 | 1735.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 1730.50 | 1735.65 | 1735.47 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2025-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 11:15:00 | 1729.90 | 1734.50 | 1734.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 12:15:00 | 1725.00 | 1732.60 | 1734.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 11:15:00 | 1721.30 | 1721.07 | 1727.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 12:00:00 | 1721.30 | 1721.07 | 1727.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 1717.00 | 1710.71 | 1716.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:00:00 | 1717.00 | 1710.71 | 1716.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 1715.20 | 1711.61 | 1716.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 1709.00 | 1711.61 | 1716.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 1723.30 | 1713.95 | 1716.80 | SL hit (close>static) qty=1.00 sl=1722.90 alert=retest2 |

### Cycle 121 — BUY (started 2025-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 11:15:00 | 1730.10 | 1713.34 | 1711.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 1749.00 | 1727.36 | 1719.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 15:15:00 | 1744.00 | 1745.50 | 1734.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 09:15:00 | 1747.30 | 1745.50 | 1734.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 1732.50 | 1742.43 | 1734.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:00:00 | 1732.50 | 1742.43 | 1734.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 1736.60 | 1741.27 | 1734.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:45:00 | 1737.20 | 1741.27 | 1734.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 1738.50 | 1740.71 | 1735.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:30:00 | 1736.60 | 1740.71 | 1735.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1737.10 | 1740.42 | 1736.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:15:00 | 1736.00 | 1740.42 | 1736.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 1746.20 | 1741.58 | 1737.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 10:30:00 | 1747.60 | 1741.40 | 1738.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 14:15:00 | 1731.80 | 1736.58 | 1737.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2025-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 14:15:00 | 1731.80 | 1736.58 | 1737.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 1718.80 | 1728.86 | 1733.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 14:15:00 | 1732.50 | 1728.53 | 1731.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 14:15:00 | 1732.50 | 1728.53 | 1731.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 1732.50 | 1728.53 | 1731.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 1732.50 | 1728.53 | 1731.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 1732.10 | 1729.25 | 1731.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 1726.00 | 1729.25 | 1731.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1720.00 | 1727.40 | 1730.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:15:00 | 1715.00 | 1727.40 | 1730.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 13:15:00 | 1737.50 | 1729.00 | 1728.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2025-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 13:15:00 | 1737.50 | 1729.00 | 1728.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 10:15:00 | 1753.10 | 1736.21 | 1731.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 13:15:00 | 1731.40 | 1736.30 | 1733.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 13:15:00 | 1731.40 | 1736.30 | 1733.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 1731.40 | 1736.30 | 1733.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:00:00 | 1731.40 | 1736.30 | 1733.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 1726.50 | 1734.34 | 1732.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:45:00 | 1718.20 | 1734.34 | 1732.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 1725.00 | 1732.47 | 1731.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 1723.80 | 1732.47 | 1731.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 1706.80 | 1727.34 | 1729.57 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 12:15:00 | 1751.60 | 1731.58 | 1728.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 11:15:00 | 1761.70 | 1742.02 | 1735.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 14:15:00 | 1743.90 | 1747.13 | 1739.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 14:15:00 | 1743.90 | 1747.13 | 1739.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 1743.90 | 1747.13 | 1739.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:45:00 | 1742.10 | 1747.13 | 1739.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 1727.90 | 1743.28 | 1738.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 1731.40 | 1743.28 | 1738.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 1730.00 | 1740.62 | 1738.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:30:00 | 1735.90 | 1739.46 | 1737.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 11:00:00 | 1734.80 | 1739.46 | 1737.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 11:45:00 | 1738.40 | 1740.63 | 1738.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 14:15:00 | 1746.00 | 1738.35 | 1737.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 1742.90 | 1739.26 | 1738.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:30:00 | 1761.90 | 1742.74 | 1740.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 11:00:00 | 1755.00 | 1745.20 | 1741.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 09:30:00 | 1754.80 | 1762.21 | 1752.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 12:15:00 | 1836.60 | 1842.80 | 1843.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 1836.60 | 1842.80 | 1843.64 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 14:15:00 | 1853.00 | 1845.40 | 1844.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 15:15:00 | 1855.00 | 1847.32 | 1845.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 10:15:00 | 1846.60 | 1847.54 | 1846.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 10:15:00 | 1846.60 | 1847.54 | 1846.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 1846.60 | 1847.54 | 1846.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:00:00 | 1846.60 | 1847.54 | 1846.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 1839.20 | 1845.87 | 1845.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:45:00 | 1835.90 | 1845.87 | 1845.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 1847.60 | 1846.22 | 1845.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 14:00:00 | 1852.70 | 1847.52 | 1846.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 10:15:00 | 1852.70 | 1847.90 | 1846.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 12:00:00 | 1852.80 | 1849.46 | 1847.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 12:45:00 | 1852.80 | 1850.47 | 1848.37 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 1852.00 | 1851.97 | 1849.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 1851.30 | 1851.97 | 1849.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1855.70 | 1852.72 | 1850.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 11:45:00 | 1874.00 | 1857.90 | 1853.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 12:45:00 | 1872.70 | 1860.78 | 1854.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 14:45:00 | 1873.80 | 1865.16 | 1857.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 1873.50 | 1865.15 | 1858.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 1886.40 | 1869.40 | 1861.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 10:45:00 | 1896.50 | 1875.30 | 1864.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 12:00:00 | 1892.70 | 1878.78 | 1867.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 12:15:00 | 1906.80 | 1927.23 | 1927.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 1906.80 | 1927.23 | 1927.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 1901.80 | 1922.15 | 1925.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 12:15:00 | 1911.80 | 1911.29 | 1917.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 12:15:00 | 1911.80 | 1911.29 | 1917.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 1911.80 | 1911.29 | 1917.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 13:00:00 | 1911.80 | 1911.29 | 1917.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 13:15:00 | 1902.20 | 1909.47 | 1915.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 13:30:00 | 1909.70 | 1909.47 | 1915.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 1879.00 | 1903.13 | 1911.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 10:45:00 | 1869.00 | 1896.01 | 1907.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:30:00 | 1874.00 | 1884.22 | 1891.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:30:00 | 1877.00 | 1872.91 | 1880.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 15:15:00 | 1775.55 | 1803.92 | 1827.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 15:15:00 | 1780.30 | 1803.92 | 1827.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 15:15:00 | 1783.15 | 1803.92 | 1827.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 1787.00 | 1764.74 | 1788.47 | SL hit (close>ema200) qty=0.50 sl=1764.74 alert=retest2 |

### Cycle 129 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 1684.40 | 1663.87 | 1661.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 1728.50 | 1682.06 | 1671.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1706.90 | 1708.82 | 1693.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 1706.90 | 1708.82 | 1693.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 1706.90 | 1708.82 | 1693.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 11:00:00 | 1725.70 | 1712.20 | 1696.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 10:45:00 | 1720.80 | 1713.58 | 1704.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 1743.90 | 1765.17 | 1767.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 1743.90 | 1765.17 | 1767.04 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2026-02-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 15:15:00 | 1774.90 | 1759.18 | 1758.10 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 10:15:00 | 1749.90 | 1756.29 | 1756.90 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 1767.00 | 1759.11 | 1758.06 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 10:15:00 | 1744.40 | 1756.00 | 1757.17 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 14:15:00 | 1769.20 | 1757.31 | 1757.14 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 1747.00 | 1755.86 | 1756.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 10:15:00 | 1736.20 | 1751.92 | 1754.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-19 12:15:00 | 1754.50 | 1751.04 | 1753.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 12:15:00 | 1754.50 | 1751.04 | 1753.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 1754.50 | 1751.04 | 1753.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:00:00 | 1754.50 | 1751.04 | 1753.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 1753.80 | 1751.60 | 1753.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:30:00 | 1755.90 | 1751.60 | 1753.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 1743.50 | 1749.98 | 1752.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 1725.90 | 1750.16 | 1752.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 09:30:00 | 1739.90 | 1735.51 | 1741.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 15:15:00 | 1652.90 | 1677.33 | 1689.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1639.61 | 1670.91 | 1685.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 15:15:00 | 1620.00 | 1618.70 | 1639.07 | SL hit (close>ema200) qty=0.50 sl=1618.70 alert=retest2 |

### Cycle 137 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 1585.20 | 1558.35 | 1555.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 14:15:00 | 1595.10 | 1565.70 | 1558.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1582.60 | 1599.70 | 1585.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1582.60 | 1599.70 | 1585.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1582.60 | 1599.70 | 1585.33 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 09:15:00 | 1570.10 | 1579.95 | 1581.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 10:15:00 | 1553.50 | 1574.66 | 1578.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1509.70 | 1493.70 | 1513.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 13:00:00 | 1509.70 | 1493.70 | 1513.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1515.90 | 1498.14 | 1513.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 1515.70 | 1498.14 | 1513.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 1502.50 | 1499.01 | 1512.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:30:00 | 1510.30 | 1499.01 | 1512.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1571.70 | 1515.15 | 1517.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 1571.70 | 1515.15 | 1517.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 1577.30 | 1527.58 | 1523.10 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 1518.30 | 1530.38 | 1531.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 1502.20 | 1524.75 | 1529.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 11:15:00 | 1519.00 | 1512.61 | 1520.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 11:15:00 | 1519.00 | 1512.61 | 1520.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 11:15:00 | 1519.00 | 1512.61 | 1520.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 12:00:00 | 1519.00 | 1512.61 | 1520.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 12:15:00 | 1517.20 | 1513.53 | 1520.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 13:00:00 | 1517.20 | 1513.53 | 1520.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 13:15:00 | 1512.30 | 1513.28 | 1519.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 13:30:00 | 1515.70 | 1513.28 | 1519.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 1500.00 | 1509.89 | 1516.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 1517.20 | 1509.89 | 1516.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1523.10 | 1512.53 | 1517.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:30:00 | 1501.50 | 1513.20 | 1516.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 1476.10 | 1514.74 | 1516.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 15:15:00 | 1524.90 | 1514.64 | 1514.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 1524.90 | 1514.64 | 1514.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 1538.50 | 1522.80 | 1518.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 1695.50 | 1702.18 | 1668.96 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 1719.00 | 1702.18 | 1668.96 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1749.30 | 1751.59 | 1718.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:00:00 | 1761.50 | 1753.91 | 1727.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 14:00:00 | 1761.00 | 1755.33 | 1730.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 1765.00 | 1752.46 | 1733.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:45:00 | 1763.50 | 1753.39 | 1735.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 09:15:00 | 1804.95 | 1771.16 | 1755.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-16 11:15:00 | 1765.00 | 1770.09 | 1757.87 | SL hit (close<ema200) qty=0.50 sl=1770.09 alert=retest1 |

### Cycle 142 — SELL (started 2026-04-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 14:15:00 | 1768.30 | 1775.17 | 1775.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 15:15:00 | 1766.60 | 1773.46 | 1774.63 | Break + close below crossover candle low |

### Cycle 143 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 1808.70 | 1780.50 | 1777.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 10:15:00 | 1832.60 | 1790.92 | 1782.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 09:15:00 | 1805.10 | 1806.17 | 1796.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-22 09:45:00 | 1803.10 | 1806.17 | 1796.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 1793.20 | 1806.69 | 1801.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:30:00 | 1794.90 | 1806.69 | 1801.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 1787.30 | 1802.81 | 1800.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:45:00 | 1788.80 | 1802.81 | 1800.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 13:15:00 | 1797.20 | 1800.16 | 1799.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 14:00:00 | 1797.20 | 1800.16 | 1799.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — SELL (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 14:15:00 | 1789.20 | 1797.97 | 1798.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 15:15:00 | 1780.00 | 1794.38 | 1797.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 14:15:00 | 1774.10 | 1770.98 | 1782.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 15:00:00 | 1774.10 | 1770.98 | 1782.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 1776.60 | 1772.11 | 1781.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 1787.90 | 1772.11 | 1781.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1786.50 | 1774.99 | 1782.12 | EMA400 retest candle locked (from downside) |

### Cycle 145 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 1792.90 | 1786.80 | 1786.14 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 09:15:00 | 1756.70 | 1782.27 | 1784.29 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 1818.50 | 1784.12 | 1781.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 12:15:00 | 1834.10 | 1794.11 | 1786.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 14:15:00 | 1791.50 | 1795.65 | 1788.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 15:00:00 | 1791.50 | 1795.65 | 1788.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 1793.00 | 1795.12 | 1788.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 1733.40 | 1795.12 | 1788.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 1739.60 | 1784.02 | 1784.41 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 1790.80 | 1779.43 | 1778.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 10:15:00 | 1812.40 | 1798.09 | 1791.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 15:15:00 | 1825.20 | 1825.60 | 1815.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 09:15:00 | 1824.80 | 1825.60 | 1815.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1824.70 | 1825.42 | 1816.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 14:30:00 | 1841.00 | 1826.80 | 1820.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 15:15:00 | 1845.00 | 1826.80 | 1820.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-14 14:15:00 | 1495.73 | 2024-05-15 12:15:00 | 1466.20 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest1 | 2024-05-15 09:15:00 | 1518.45 | 2024-05-15 12:15:00 | 1466.20 | STOP_HIT | 1.00 | -3.44% |
| BUY | retest1 | 2024-05-15 11:00:00 | 1494.98 | 2024-05-15 12:15:00 | 1466.20 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest1 | 2024-05-15 11:45:00 | 1499.70 | 2024-05-15 12:15:00 | 1466.20 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2024-05-17 12:00:00 | 1503.68 | 2024-05-22 09:15:00 | 1654.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-17 15:00:00 | 1520.50 | 2024-05-22 09:15:00 | 1652.78 | TARGET_HIT | 1.00 | 8.70% |
| BUY | retest2 | 2024-05-21 11:00:00 | 1502.53 | 2024-05-28 09:15:00 | 1521.63 | STOP_HIT | 1.00 | 1.27% |
| SELL | retest2 | 2024-05-31 10:15:00 | 1527.28 | 2024-05-31 12:15:00 | 1586.55 | STOP_HIT | 1.00 | -3.88% |
| SELL | retest2 | 2024-05-31 11:15:00 | 1521.30 | 2024-05-31 12:15:00 | 1586.55 | STOP_HIT | 1.00 | -4.29% |
| BUY | retest2 | 2024-06-04 15:00:00 | 1647.03 | 2024-06-05 09:15:00 | 1500.70 | STOP_HIT | 1.00 | -8.88% |
| BUY | retest2 | 2024-06-13 11:15:00 | 1701.08 | 2024-06-18 14:15:00 | 1871.19 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-25 09:15:00 | 1830.73 | 2024-06-27 12:15:00 | 1794.90 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-07-24 11:00:00 | 1802.05 | 2024-07-29 14:15:00 | 1815.60 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-07-24 13:00:00 | 1802.60 | 2024-07-29 14:15:00 | 1815.60 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-07-24 13:45:00 | 1792.65 | 2024-07-29 14:15:00 | 1815.60 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-07-29 10:15:00 | 1802.33 | 2024-07-29 14:15:00 | 1815.60 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-08-02 09:15:00 | 1785.55 | 2024-08-05 09:15:00 | 1606.99 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-02 09:45:00 | 1782.05 | 2024-08-05 09:15:00 | 1603.85 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-19 09:15:00 | 1726.03 | 2024-08-30 14:15:00 | 1898.63 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-19 11:15:00 | 1710.00 | 2024-08-30 14:15:00 | 1881.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-19 12:00:00 | 1712.10 | 2024-08-30 14:15:00 | 1883.31 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-09-05 09:15:00 | 1804.90 | 2024-09-06 09:15:00 | 1868.13 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2024-09-05 11:15:00 | 1806.88 | 2024-09-06 09:15:00 | 1868.13 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2024-09-11 10:00:00 | 1760.65 | 2024-09-12 15:15:00 | 1672.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-11 10:00:00 | 1760.65 | 2024-09-16 09:15:00 | 1713.93 | STOP_HIT | 0.50 | 2.65% |
| SELL | retest2 | 2024-10-07 15:15:00 | 1661.00 | 2024-10-09 09:15:00 | 1749.65 | STOP_HIT | 1.00 | -5.34% |
| SELL | retest2 | 2024-10-08 13:15:00 | 1655.00 | 2024-10-09 09:15:00 | 1749.65 | STOP_HIT | 1.00 | -5.72% |
| SELL | retest2 | 2024-10-23 09:15:00 | 1524.45 | 2024-10-25 10:15:00 | 1487.08 | PARTIAL | 0.50 | 2.45% |
| SELL | retest2 | 2024-10-23 15:00:00 | 1565.35 | 2024-10-25 10:15:00 | 1479.58 | PARTIAL | 0.50 | 5.48% |
| SELL | retest2 | 2024-10-23 09:15:00 | 1524.45 | 2024-10-28 09:15:00 | 1372.01 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-23 15:00:00 | 1565.35 | 2024-10-28 09:15:00 | 1408.82 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-24 09:15:00 | 1557.45 | 2024-10-28 09:15:00 | 1401.71 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-06 10:30:00 | 1497.85 | 2024-11-06 11:15:00 | 1545.50 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2024-11-21 12:15:00 | 1504.95 | 2024-11-25 09:15:00 | 1655.45 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-03 10:30:00 | 1586.65 | 2025-01-08 13:15:00 | 1647.00 | STOP_HIT | 1.00 | -3.80% |
| BUY | retest2 | 2025-01-10 09:15:00 | 1661.25 | 2025-01-13 09:15:00 | 1587.65 | STOP_HIT | 1.00 | -4.43% |
| BUY | retest2 | 2025-01-10 13:30:00 | 1654.80 | 2025-01-13 09:15:00 | 1587.65 | STOP_HIT | 1.00 | -4.06% |
| BUY | retest2 | 2025-01-10 14:00:00 | 1654.05 | 2025-01-13 09:15:00 | 1587.65 | STOP_HIT | 1.00 | -4.01% |
| BUY | retest2 | 2025-01-17 12:45:00 | 1667.60 | 2025-01-20 09:15:00 | 1627.00 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-01-23 12:15:00 | 1507.05 | 2025-01-23 14:15:00 | 1530.15 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-01-23 13:15:00 | 1514.50 | 2025-01-23 14:15:00 | 1530.15 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-01-24 10:00:00 | 1508.85 | 2025-01-24 13:15:00 | 1530.55 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-01-27 09:15:00 | 1498.05 | 2025-01-28 15:15:00 | 1517.45 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-01-27 12:00:00 | 1481.60 | 2025-01-28 15:15:00 | 1517.45 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-01-27 12:30:00 | 1478.50 | 2025-01-28 15:15:00 | 1517.45 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2025-02-13 09:15:00 | 1591.45 | 2025-02-13 12:15:00 | 1608.50 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-02-17 09:15:00 | 1550.00 | 2025-02-18 13:15:00 | 1586.20 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2025-02-21 12:15:00 | 1584.15 | 2025-02-24 09:15:00 | 1562.75 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-02-21 14:15:00 | 1589.45 | 2025-02-24 09:15:00 | 1562.75 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-02-21 15:00:00 | 1590.35 | 2025-02-24 09:15:00 | 1562.75 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-02-25 09:15:00 | 1530.60 | 2025-02-25 12:15:00 | 1580.00 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2025-02-25 12:30:00 | 1567.85 | 2025-02-28 09:15:00 | 1489.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 14:15:00 | 1566.95 | 2025-02-28 09:15:00 | 1488.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 12:30:00 | 1567.85 | 2025-02-28 13:15:00 | 1527.80 | STOP_HIT | 0.50 | 2.55% |
| SELL | retest2 | 2025-02-25 14:15:00 | 1566.95 | 2025-02-28 13:15:00 | 1527.80 | STOP_HIT | 0.50 | 2.50% |
| BUY | retest2 | 2025-03-04 14:45:00 | 1558.90 | 2025-03-10 15:15:00 | 1551.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-03-04 15:15:00 | 1558.00 | 2025-03-10 15:15:00 | 1551.00 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-03-05 11:30:00 | 1567.20 | 2025-03-10 15:15:00 | 1551.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-03-26 09:15:00 | 1678.40 | 2025-03-26 10:15:00 | 1647.20 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-03-28 09:15:00 | 1644.95 | 2025-03-28 09:15:00 | 1662.20 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-04-04 10:15:00 | 1591.40 | 2025-04-07 09:15:00 | 1511.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 12:15:00 | 1594.15 | 2025-04-07 09:15:00 | 1514.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 12:45:00 | 1592.05 | 2025-04-07 09:15:00 | 1512.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 10:15:00 | 1591.40 | 2025-04-08 09:15:00 | 1569.00 | STOP_HIT | 0.50 | 1.41% |
| SELL | retest2 | 2025-04-04 12:15:00 | 1594.15 | 2025-04-08 09:15:00 | 1569.00 | STOP_HIT | 0.50 | 1.58% |
| SELL | retest2 | 2025-04-04 12:45:00 | 1592.05 | 2025-04-08 09:15:00 | 1569.00 | STOP_HIT | 0.50 | 1.45% |
| SELL | retest2 | 2025-04-29 13:15:00 | 1632.40 | 2025-04-30 09:15:00 | 1672.70 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-04-29 14:45:00 | 1636.60 | 2025-04-30 09:15:00 | 1672.70 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-05-08 13:30:00 | 1498.90 | 2025-05-12 13:15:00 | 1531.30 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-05-12 09:30:00 | 1510.00 | 2025-05-12 13:15:00 | 1531.30 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-05-12 10:45:00 | 1513.10 | 2025-05-12 13:15:00 | 1531.30 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-05-15 13:00:00 | 1546.30 | 2025-05-23 15:15:00 | 1590.00 | STOP_HIT | 1.00 | 2.83% |
| BUY | retest2 | 2025-05-16 11:15:00 | 1549.80 | 2025-05-23 15:15:00 | 1590.00 | STOP_HIT | 1.00 | 2.59% |
| BUY | retest2 | 2025-05-28 12:30:00 | 1604.50 | 2025-05-28 13:15:00 | 1596.00 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2025-07-08 09:15:00 | 1559.90 | 2025-07-09 09:15:00 | 1517.40 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2025-07-14 11:30:00 | 1501.30 | 2025-07-14 14:15:00 | 1512.20 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-07-14 13:00:00 | 1501.50 | 2025-07-14 14:15:00 | 1512.20 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-07-15 09:30:00 | 1501.20 | 2025-07-17 10:15:00 | 1511.10 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-07-17 09:30:00 | 1499.90 | 2025-07-17 10:15:00 | 1511.10 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-07-22 12:30:00 | 1465.40 | 2025-07-25 09:15:00 | 1522.60 | STOP_HIT | 1.00 | -3.90% |
| SELL | retest2 | 2025-07-23 09:30:00 | 1461.00 | 2025-07-25 09:15:00 | 1522.60 | STOP_HIT | 1.00 | -4.22% |
| BUY | retest1 | 2025-07-28 10:45:00 | 1524.60 | 2025-07-29 09:15:00 | 1492.60 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest1 | 2025-07-28 11:15:00 | 1524.10 | 2025-07-29 09:15:00 | 1492.60 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-07-29 15:15:00 | 1511.50 | 2025-07-30 15:15:00 | 1499.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-07-30 11:15:00 | 1513.90 | 2025-07-30 15:15:00 | 1499.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-07-30 11:45:00 | 1511.10 | 2025-07-30 15:15:00 | 1499.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-07-30 14:30:00 | 1512.20 | 2025-07-30 15:15:00 | 1499.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-08-06 09:15:00 | 1456.00 | 2025-08-13 09:15:00 | 1453.30 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-09-15 13:15:00 | 1596.70 | 2025-09-23 14:15:00 | 1626.70 | STOP_HIT | 1.00 | 1.88% |
| BUY | retest2 | 2025-09-16 09:15:00 | 1598.40 | 2025-09-23 14:15:00 | 1626.70 | STOP_HIT | 1.00 | 1.77% |
| SELL | retest2 | 2025-09-30 09:15:00 | 1541.90 | 2025-10-01 12:15:00 | 1572.00 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-10-09 12:30:00 | 1592.50 | 2025-10-20 14:15:00 | 1657.80 | STOP_HIT | 1.00 | 4.10% |
| BUY | retest2 | 2025-10-09 15:00:00 | 1595.20 | 2025-10-20 14:15:00 | 1657.80 | STOP_HIT | 1.00 | 3.92% |
| BUY | retest2 | 2025-10-10 09:30:00 | 1603.60 | 2025-10-20 14:15:00 | 1657.80 | STOP_HIT | 1.00 | 3.38% |
| BUY | retest2 | 2025-10-13 09:45:00 | 1593.90 | 2025-10-20 14:15:00 | 1657.80 | STOP_HIT | 1.00 | 4.01% |
| BUY | retest2 | 2025-10-13 12:00:00 | 1618.90 | 2025-10-20 14:15:00 | 1657.80 | STOP_HIT | 1.00 | 2.40% |
| BUY | retest2 | 2025-11-07 11:45:00 | 1770.10 | 2025-11-11 10:15:00 | 1740.90 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-11-07 14:15:00 | 1773.80 | 2025-11-11 10:15:00 | 1740.90 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-11-14 14:45:00 | 1729.00 | 2025-11-17 10:15:00 | 1742.70 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-11-17 09:15:00 | 1727.80 | 2025-11-17 10:15:00 | 1742.70 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-11-21 09:15:00 | 1709.00 | 2025-11-21 09:15:00 | 1723.30 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-11-21 11:15:00 | 1713.10 | 2025-11-25 09:15:00 | 1724.60 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-11-21 12:45:00 | 1708.00 | 2025-11-25 09:15:00 | 1724.60 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-11-24 10:45:00 | 1710.10 | 2025-11-25 09:15:00 | 1724.60 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-11-24 14:45:00 | 1691.00 | 2025-11-25 09:15:00 | 1724.60 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-12-01 10:30:00 | 1747.60 | 2025-12-01 14:15:00 | 1731.80 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-12-03 10:15:00 | 1715.00 | 2025-12-04 13:15:00 | 1737.50 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-12-11 10:30:00 | 1735.90 | 2025-12-29 12:15:00 | 1836.60 | STOP_HIT | 1.00 | 5.80% |
| BUY | retest2 | 2025-12-11 11:00:00 | 1734.80 | 2025-12-29 12:15:00 | 1836.60 | STOP_HIT | 1.00 | 5.87% |
| BUY | retest2 | 2025-12-11 11:45:00 | 1738.40 | 2025-12-29 12:15:00 | 1836.60 | STOP_HIT | 1.00 | 5.65% |
| BUY | retest2 | 2025-12-11 14:15:00 | 1746.00 | 2025-12-29 12:15:00 | 1836.60 | STOP_HIT | 1.00 | 5.19% |
| BUY | retest2 | 2025-12-12 09:30:00 | 1761.90 | 2025-12-29 12:15:00 | 1836.60 | STOP_HIT | 1.00 | 4.24% |
| BUY | retest2 | 2025-12-12 11:00:00 | 1755.00 | 2025-12-29 12:15:00 | 1836.60 | STOP_HIT | 1.00 | 4.65% |
| BUY | retest2 | 2025-12-15 09:30:00 | 1754.80 | 2025-12-29 12:15:00 | 1836.60 | STOP_HIT | 1.00 | 4.66% |
| BUY | retest2 | 2025-12-30 14:00:00 | 1852.70 | 2026-01-08 12:15:00 | 1906.80 | STOP_HIT | 1.00 | 2.92% |
| BUY | retest2 | 2025-12-31 10:15:00 | 1852.70 | 2026-01-08 12:15:00 | 1906.80 | STOP_HIT | 1.00 | 2.92% |
| BUY | retest2 | 2025-12-31 12:00:00 | 1852.80 | 2026-01-08 12:15:00 | 1906.80 | STOP_HIT | 1.00 | 2.91% |
| BUY | retest2 | 2025-12-31 12:45:00 | 1852.80 | 2026-01-08 12:15:00 | 1906.80 | STOP_HIT | 1.00 | 2.91% |
| BUY | retest2 | 2026-01-01 11:45:00 | 1874.00 | 2026-01-08 12:15:00 | 1906.80 | STOP_HIT | 1.00 | 1.75% |
| BUY | retest2 | 2026-01-01 12:45:00 | 1872.70 | 2026-01-08 12:15:00 | 1906.80 | STOP_HIT | 1.00 | 1.82% |
| BUY | retest2 | 2026-01-01 14:45:00 | 1873.80 | 2026-01-08 12:15:00 | 1906.80 | STOP_HIT | 1.00 | 1.76% |
| BUY | retest2 | 2026-01-02 09:15:00 | 1873.50 | 2026-01-08 12:15:00 | 1906.80 | STOP_HIT | 1.00 | 1.78% |
| BUY | retest2 | 2026-01-02 10:45:00 | 1896.50 | 2026-01-08 12:15:00 | 1906.80 | STOP_HIT | 1.00 | 0.54% |
| BUY | retest2 | 2026-01-02 12:00:00 | 1892.70 | 2026-01-08 12:15:00 | 1906.80 | STOP_HIT | 1.00 | 0.74% |
| SELL | retest2 | 2026-01-12 10:45:00 | 1869.00 | 2026-01-20 15:15:00 | 1775.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 09:30:00 | 1874.00 | 2026-01-20 15:15:00 | 1780.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 09:30:00 | 1877.00 | 2026-01-20 15:15:00 | 1783.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-12 10:45:00 | 1869.00 | 2026-01-22 09:15:00 | 1787.00 | STOP_HIT | 0.50 | 4.39% |
| SELL | retest2 | 2026-01-14 09:30:00 | 1874.00 | 2026-01-22 09:15:00 | 1787.00 | STOP_HIT | 0.50 | 4.64% |
| SELL | retest2 | 2026-01-16 09:30:00 | 1877.00 | 2026-01-22 09:15:00 | 1787.00 | STOP_HIT | 0.50 | 4.79% |
| BUY | retest2 | 2026-02-05 11:00:00 | 1725.70 | 2026-02-13 10:15:00 | 1743.90 | STOP_HIT | 1.00 | 1.05% |
| BUY | retest2 | 2026-02-06 10:45:00 | 1720.80 | 2026-02-13 10:15:00 | 1743.90 | STOP_HIT | 1.00 | 1.34% |
| SELL | retest2 | 2026-02-20 09:15:00 | 1725.90 | 2026-02-27 15:15:00 | 1652.90 | PARTIAL | 0.50 | 4.23% |
| SELL | retest2 | 2026-02-23 09:30:00 | 1739.90 | 2026-03-02 09:15:00 | 1639.61 | PARTIAL | 0.50 | 5.76% |
| SELL | retest2 | 2026-02-20 09:15:00 | 1725.90 | 2026-03-04 15:15:00 | 1620.00 | STOP_HIT | 0.50 | 6.14% |
| SELL | retest2 | 2026-02-23 09:30:00 | 1739.90 | 2026-03-04 15:15:00 | 1620.00 | STOP_HIT | 0.50 | 6.89% |
| SELL | retest2 | 2026-04-01 11:30:00 | 1501.50 | 2026-04-02 15:15:00 | 1524.90 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2026-04-02 09:15:00 | 1476.10 | 2026-04-02 15:15:00 | 1524.90 | STOP_HIT | 1.00 | -3.31% |
| BUY | retest1 | 2026-04-10 09:15:00 | 1719.00 | 2026-04-16 09:15:00 | 1804.95 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-04-10 09:15:00 | 1719.00 | 2026-04-16 11:15:00 | 1765.00 | STOP_HIT | 0.50 | 2.68% |
| BUY | retest2 | 2026-04-13 13:00:00 | 1761.50 | 2026-04-20 14:15:00 | 1768.30 | STOP_HIT | 1.00 | 0.39% |
| BUY | retest2 | 2026-04-13 14:00:00 | 1761.00 | 2026-04-20 14:15:00 | 1768.30 | STOP_HIT | 1.00 | 0.41% |
| BUY | retest2 | 2026-04-15 09:15:00 | 1765.00 | 2026-04-20 14:15:00 | 1768.30 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2026-04-15 09:45:00 | 1763.50 | 2026-04-20 14:15:00 | 1768.30 | STOP_HIT | 1.00 | 0.27% |
