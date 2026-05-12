# Data Patterns (India) Ltd. (DATAPATTNS)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 4118.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 190 |
| ALERT1 | 135 |
| ALERT2 | 135 |
| ALERT2_SKIP | 66 |
| ALERT3 | 364 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 186 |
| PARTIAL | 32 |
| TARGET_HIT | 32 |
| STOP_HIT | 159 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 223 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 99 / 124
- **Target hits / Stop hits / Partials:** 32 / 159 / 32
- **Avg / median % per leg:** 1.46% / -0.47%
- **Sum % (uncompounded):** 324.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 76 | 26 | 34.2% | 18 | 58 | 0 | 1.28% | 97.4% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.49% | -5.9% |
| BUY @ 3rd Alert (retest2) | 72 | 26 | 36.1% | 18 | 54 | 0 | 1.43% | 103.3% |
| SELL (all) | 147 | 73 | 49.7% | 14 | 101 | 32 | 1.55% | 227.4% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| SELL @ 3rd Alert (retest2) | 146 | 72 | 49.3% | 13 | 101 | 32 | 1.49% | 217.4% |
| retest1 (combined) | 5 | 1 | 20.0% | 1 | 4 | 0 | 0.81% | 4.1% |
| retest2 (combined) | 218 | 98 | 45.0% | 31 | 155 | 32 | 1.47% | 320.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-17 09:15:00 | 1622.65 | 1592.16 | 1589.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-17 15:15:00 | 1648.00 | 1624.58 | 1609.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-18 10:15:00 | 1617.15 | 1625.01 | 1612.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-18 10:15:00 | 1617.15 | 1625.01 | 1612.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 10:15:00 | 1617.15 | 1625.01 | 1612.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-18 10:30:00 | 1617.80 | 1625.01 | 1612.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 13:15:00 | 1614.00 | 1622.20 | 1614.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-18 14:00:00 | 1614.00 | 1622.20 | 1614.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 14:15:00 | 1608.85 | 1619.53 | 1613.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-18 15:00:00 | 1608.85 | 1619.53 | 1613.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 15:15:00 | 1608.60 | 1617.35 | 1613.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-19 09:15:00 | 1618.65 | 1617.35 | 1613.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-19 10:15:00 | 1596.00 | 1610.43 | 1610.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2023-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 10:15:00 | 1596.00 | 1610.43 | 1610.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-24 12:15:00 | 1590.00 | 1595.21 | 1598.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-24 14:15:00 | 1600.85 | 1592.31 | 1596.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-24 14:15:00 | 1600.85 | 1592.31 | 1596.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 14:15:00 | 1600.85 | 1592.31 | 1596.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-24 15:00:00 | 1600.85 | 1592.31 | 1596.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 15:15:00 | 1600.90 | 1594.03 | 1596.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-25 09:15:00 | 1576.95 | 1594.03 | 1596.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-25 14:15:00 | 1603.00 | 1588.13 | 1591.08 | SL hit (close>static) qty=1.00 sl=1601.00 alert=retest2 |

### Cycle 3 — BUY (started 2023-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 09:15:00 | 1614.60 | 1595.29 | 1593.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-29 09:15:00 | 1652.00 | 1614.05 | 1605.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 14:15:00 | 1665.00 | 1676.00 | 1655.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-30 14:30:00 | 1678.65 | 1676.00 | 1655.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 15:15:00 | 1661.70 | 1673.14 | 1655.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 09:30:00 | 1671.70 | 1678.11 | 1659.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 10:00:00 | 1698.00 | 1678.11 | 1659.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-01 09:45:00 | 1675.60 | 1679.18 | 1669.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2023-06-07 09:15:00 | 1838.87 | 1779.63 | 1754.73 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2023-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-12 13:15:00 | 1756.15 | 1774.04 | 1775.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-12 14:15:00 | 1742.60 | 1767.75 | 1772.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-14 09:15:00 | 1729.50 | 1722.73 | 1741.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-14 10:00:00 | 1729.50 | 1722.73 | 1741.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 10:15:00 | 1733.55 | 1724.89 | 1740.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-14 10:30:00 | 1743.25 | 1724.89 | 1740.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 12:15:00 | 1754.15 | 1732.60 | 1741.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-14 13:00:00 | 1754.15 | 1732.60 | 1741.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 13:15:00 | 1724.00 | 1730.88 | 1740.09 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2023-06-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-15 10:15:00 | 1781.95 | 1749.11 | 1746.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-15 12:15:00 | 1792.00 | 1763.47 | 1753.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-20 09:15:00 | 1894.80 | 1910.47 | 1875.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 09:15:00 | 1894.80 | 1910.47 | 1875.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 09:15:00 | 1894.80 | 1910.47 | 1875.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 09:30:00 | 1887.15 | 1910.47 | 1875.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 11:15:00 | 1892.00 | 1902.87 | 1877.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-20 12:45:00 | 1893.95 | 1900.83 | 1879.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-20 13:30:00 | 1895.75 | 1898.27 | 1879.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-20 15:15:00 | 1897.50 | 1896.45 | 1880.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-21 11:15:00 | 1902.65 | 1898.63 | 1885.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 11:15:00 | 1910.95 | 1901.10 | 1888.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-21 12:30:00 | 1916.00 | 1903.28 | 1890.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-21 13:30:00 | 1918.45 | 1905.27 | 1892.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-21 15:15:00 | 1915.00 | 1904.79 | 1893.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-22 11:15:00 | 1866.25 | 1895.32 | 1892.62 | SL hit (close<static) qty=1.00 sl=1876.00 alert=retest2 |

### Cycle 6 — SELL (started 2023-06-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 12:15:00 | 1842.00 | 1884.66 | 1888.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 13:15:00 | 1839.00 | 1875.53 | 1883.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 09:15:00 | 1819.45 | 1819.44 | 1840.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-26 09:45:00 | 1807.35 | 1819.44 | 1840.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 1846.50 | 1819.56 | 1829.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 10:00:00 | 1846.50 | 1819.56 | 1829.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 1848.45 | 1825.34 | 1831.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 12:45:00 | 1842.45 | 1832.70 | 1833.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-27 13:15:00 | 1844.00 | 1834.96 | 1834.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2023-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 13:15:00 | 1844.00 | 1834.96 | 1834.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 14:15:00 | 1869.75 | 1841.92 | 1837.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 14:15:00 | 1863.95 | 1868.43 | 1856.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-28 15:00:00 | 1863.95 | 1868.43 | 1856.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 13:15:00 | 1869.50 | 1873.48 | 1865.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-03 09:15:00 | 1887.65 | 1871.58 | 1865.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-03 11:15:00 | 1882.90 | 1873.39 | 1867.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-04 10:30:00 | 1877.45 | 1879.79 | 1874.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-04 15:15:00 | 1880.25 | 1879.18 | 1876.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 15:15:00 | 1880.25 | 1879.39 | 1876.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-05 09:15:00 | 1882.50 | 1879.39 | 1876.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 1875.40 | 1878.59 | 1876.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-05 09:30:00 | 1879.05 | 1878.59 | 1876.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 10:15:00 | 1876.60 | 1878.19 | 1876.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-05 11:15:00 | 1882.00 | 1878.19 | 1876.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-10 09:15:00 | 1880.55 | 1896.56 | 1897.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2023-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 09:15:00 | 1880.55 | 1896.56 | 1897.07 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 12:15:00 | 1940.00 | 1903.93 | 1900.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 09:15:00 | 1971.00 | 1920.95 | 1909.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-11 15:15:00 | 1940.85 | 1951.40 | 1933.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-12 09:15:00 | 1932.55 | 1951.40 | 1933.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 09:15:00 | 2004.40 | 1962.00 | 1939.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-12 15:15:00 | 2045.00 | 1992.15 | 1965.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-07-13 11:15:00 | 2249.50 | 2100.25 | 2030.05 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2023-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 12:15:00 | 2032.70 | 2054.68 | 2055.23 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-18 14:15:00 | 2064.05 | 2056.82 | 2056.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-20 14:15:00 | 2088.00 | 2072.24 | 2066.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-20 15:15:00 | 2070.00 | 2071.79 | 2066.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-20 15:15:00 | 2070.00 | 2071.79 | 2066.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 15:15:00 | 2070.00 | 2071.79 | 2066.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-21 09:15:00 | 2067.00 | 2071.79 | 2066.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 09:15:00 | 2073.80 | 2072.19 | 2067.10 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 15:15:00 | 2059.00 | 2064.37 | 2065.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-24 09:15:00 | 2013.00 | 2054.09 | 2060.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-26 09:15:00 | 2012.00 | 2007.48 | 2021.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-26 09:15:00 | 2012.00 | 2007.48 | 2021.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 09:15:00 | 2012.00 | 2007.48 | 2021.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-27 09:15:00 | 1987.50 | 2008.67 | 2016.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-27 10:00:00 | 1987.65 | 2004.47 | 2014.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-27 11:45:00 | 1993.90 | 2000.04 | 2010.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-27 12:45:00 | 1997.00 | 1999.42 | 2009.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 14:15:00 | 2014.30 | 1981.04 | 1990.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-28 15:00:00 | 2014.30 | 1981.04 | 1990.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 15:15:00 | 2008.00 | 1986.44 | 1991.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 09:15:00 | 2020.00 | 1986.44 | 1991.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-07-31 10:15:00 | 2012.15 | 1997.11 | 1996.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2023-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 10:15:00 | 2012.15 | 1997.11 | 1996.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 12:15:00 | 2027.50 | 2006.03 | 2000.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 09:15:00 | 2016.10 | 2018.51 | 2009.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-01 10:00:00 | 2016.10 | 2018.51 | 2009.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 10:15:00 | 2009.60 | 2016.73 | 2009.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 11:00:00 | 2009.60 | 2016.73 | 2009.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 11:15:00 | 2009.00 | 2015.18 | 2009.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 11:30:00 | 2010.60 | 2015.18 | 2009.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 12:15:00 | 2005.15 | 2013.17 | 2008.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 13:00:00 | 2005.15 | 2013.17 | 2008.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 13:15:00 | 2000.70 | 2010.68 | 2008.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 14:00:00 | 2000.70 | 2010.68 | 2008.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 14:15:00 | 2013.00 | 2011.14 | 2008.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-02 09:45:00 | 2022.40 | 2012.42 | 2009.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-02 11:15:00 | 2017.60 | 2012.46 | 2009.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-02 12:00:00 | 2019.00 | 2013.77 | 2010.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-02 13:00:00 | 2017.95 | 2014.60 | 2011.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-02 13:15:00 | 1971.90 | 2006.06 | 2007.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2023-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 13:15:00 | 1971.90 | 2006.06 | 2007.71 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 14:15:00 | 2010.60 | 1991.72 | 1990.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 15:15:00 | 2019.80 | 1997.34 | 1993.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-10 13:15:00 | 2105.40 | 2107.63 | 2079.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-10 13:45:00 | 2103.45 | 2107.63 | 2079.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 14:15:00 | 2124.00 | 2110.91 | 2083.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 14:30:00 | 2074.95 | 2110.91 | 2083.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 14:15:00 | 2086.20 | 2107.36 | 2095.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 14:45:00 | 2096.40 | 2107.36 | 2095.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 15:15:00 | 2090.00 | 2103.89 | 2095.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 09:15:00 | 2065.15 | 2103.89 | 2095.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 12:15:00 | 2095.50 | 2102.48 | 2097.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 12:45:00 | 2092.45 | 2102.48 | 2097.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 13:15:00 | 2095.65 | 2101.12 | 2097.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 13:30:00 | 2099.85 | 2101.12 | 2097.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 14:15:00 | 2099.65 | 2100.82 | 2097.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 14:30:00 | 2096.70 | 2100.82 | 2097.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 15:15:00 | 2091.00 | 2098.86 | 2096.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-16 09:15:00 | 2122.15 | 2098.86 | 2096.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-08-21 11:15:00 | 2334.37 | 2239.70 | 2206.71 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2023-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 13:15:00 | 2329.10 | 2346.68 | 2348.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 14:15:00 | 2319.15 | 2341.18 | 2345.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 09:15:00 | 2345.20 | 2340.35 | 2344.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 09:15:00 | 2345.20 | 2340.35 | 2344.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 2345.20 | 2340.35 | 2344.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 09:45:00 | 2345.25 | 2340.35 | 2344.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 10:15:00 | 2340.05 | 2340.29 | 2343.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-28 15:15:00 | 2320.00 | 2340.52 | 2342.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-29 11:00:00 | 2323.65 | 2335.70 | 2339.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-30 11:45:00 | 2313.10 | 2323.38 | 2329.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-31 11:15:00 | 2358.15 | 2296.83 | 2306.64 | SL hit (close>static) qty=1.00 sl=2349.00 alert=retest2 |

### Cycle 17 — BUY (started 2023-08-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 13:15:00 | 2360.15 | 2320.08 | 2316.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-31 14:15:00 | 2387.15 | 2333.50 | 2322.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-04 12:15:00 | 2417.35 | 2441.06 | 2408.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-04 13:00:00 | 2417.35 | 2441.06 | 2408.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 14:15:00 | 2416.00 | 2432.04 | 2409.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 15:15:00 | 2408.10 | 2432.04 | 2409.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 15:15:00 | 2408.10 | 2427.25 | 2409.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-05 09:15:00 | 2428.80 | 2427.25 | 2409.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-05 10:15:00 | 2397.00 | 2422.44 | 2410.61 | SL hit (close<static) qty=1.00 sl=2404.00 alert=retest2 |

### Cycle 18 — SELL (started 2023-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-05 14:15:00 | 2393.15 | 2403.53 | 2404.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-06 10:15:00 | 2372.30 | 2394.41 | 2399.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-11 09:15:00 | 2237.00 | 2194.48 | 2234.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-11 09:15:00 | 2237.00 | 2194.48 | 2234.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 09:15:00 | 2237.00 | 2194.48 | 2234.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-11 10:00:00 | 2237.00 | 2194.48 | 2234.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 10:15:00 | 2254.05 | 2206.40 | 2236.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-11 10:45:00 | 2267.00 | 2206.40 | 2236.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 15:15:00 | 2250.00 | 2236.03 | 2242.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:15:00 | 2225.00 | 2236.03 | 2242.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 2171.20 | 2223.06 | 2235.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-13 09:30:00 | 2065.30 | 2139.12 | 2182.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 10:00:00 | 2098.00 | 2082.93 | 2125.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-20 09:30:00 | 2114.85 | 2082.93 | 2090.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-21 15:15:00 | 2109.00 | 2085.24 | 2084.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2023-09-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-21 15:15:00 | 2109.00 | 2085.24 | 2084.88 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2023-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-22 09:15:00 | 2063.00 | 2080.79 | 2082.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-25 14:15:00 | 2060.05 | 2070.02 | 2073.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-26 09:15:00 | 2078.00 | 2070.75 | 2073.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 09:15:00 | 2078.00 | 2070.75 | 2073.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 2078.00 | 2070.75 | 2073.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 09:45:00 | 2071.20 | 2070.75 | 2073.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 10:15:00 | 2089.90 | 2074.58 | 2075.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 10:30:00 | 2095.55 | 2074.58 | 2075.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2023-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 11:15:00 | 2107.15 | 2081.09 | 2077.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 11:15:00 | 2124.45 | 2107.37 | 2100.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 11:15:00 | 2090.00 | 2111.56 | 2107.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 11:15:00 | 2090.00 | 2111.56 | 2107.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 11:15:00 | 2090.00 | 2111.56 | 2107.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 11:45:00 | 2078.80 | 2111.56 | 2107.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2023-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 12:15:00 | 2073.00 | 2103.85 | 2104.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 10:15:00 | 2070.05 | 2086.04 | 2091.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 2082.95 | 2069.38 | 2078.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 2082.95 | 2069.38 | 2078.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 2082.95 | 2069.38 | 2078.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-10 13:15:00 | 2068.90 | 2071.70 | 2077.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-11 10:15:00 | 2097.95 | 2076.92 | 2077.33 | SL hit (close>static) qty=1.00 sl=2095.05 alert=retest2 |

### Cycle 23 — BUY (started 2023-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 11:15:00 | 2090.75 | 2079.69 | 2078.55 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2023-10-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 10:15:00 | 2070.05 | 2079.83 | 2079.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-12 13:15:00 | 2063.60 | 2073.72 | 2076.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-17 09:15:00 | 2117.70 | 2029.20 | 2035.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 09:15:00 | 2117.70 | 2029.20 | 2035.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 09:15:00 | 2117.70 | 2029.20 | 2035.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 10:00:00 | 2117.70 | 2029.20 | 2035.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2023-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 10:15:00 | 2106.95 | 2044.75 | 2042.03 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-10-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 13:15:00 | 2055.65 | 2068.38 | 2068.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 12:15:00 | 2045.25 | 2062.64 | 2065.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-26 13:15:00 | 1819.65 | 1816.74 | 1879.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-26 15:15:00 | 1855.00 | 1826.17 | 1872.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 15:15:00 | 1855.00 | 1826.17 | 1872.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 09:15:00 | 1890.65 | 1826.17 | 1872.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 1914.55 | 1843.85 | 1876.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 09:30:00 | 1919.80 | 1843.85 | 1876.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 1915.00 | 1858.08 | 1880.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 11:00:00 | 1915.00 | 1858.08 | 1880.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2023-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 14:15:00 | 1919.80 | 1895.43 | 1893.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-27 15:15:00 | 1931.50 | 1902.64 | 1896.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-30 10:15:00 | 1895.65 | 1904.02 | 1898.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-30 10:15:00 | 1895.65 | 1904.02 | 1898.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 10:15:00 | 1895.65 | 1904.02 | 1898.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-30 11:00:00 | 1895.65 | 1904.02 | 1898.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 11:15:00 | 1896.40 | 1902.50 | 1898.34 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 14:15:00 | 1881.70 | 1896.49 | 1896.50 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2023-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 15:15:00 | 1900.10 | 1897.21 | 1896.83 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 10:15:00 | 1882.00 | 1894.31 | 1895.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 14:15:00 | 1875.70 | 1887.01 | 1891.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 10:15:00 | 1868.00 | 1866.53 | 1875.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-02 11:15:00 | 1872.50 | 1866.53 | 1875.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 11:15:00 | 1857.20 | 1864.66 | 1873.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 11:30:00 | 1870.25 | 1864.66 | 1873.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 14:15:00 | 1861.15 | 1860.57 | 1869.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 15:00:00 | 1861.15 | 1860.57 | 1869.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 09:15:00 | 1912.30 | 1871.32 | 1872.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-03 09:45:00 | 1921.65 | 1871.32 | 1872.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2023-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 10:15:00 | 1899.50 | 1876.96 | 1875.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 15:15:00 | 1932.00 | 1901.39 | 1888.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-06 15:15:00 | 1956.00 | 1956.31 | 1930.41 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-07 09:15:00 | 1969.70 | 1956.31 | 1930.41 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-07 11:45:00 | 1965.15 | 1961.78 | 1939.65 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 09:15:00 | 1967.70 | 1958.86 | 1945.26 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 09:45:00 | 1967.60 | 1963.89 | 1948.78 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 13:15:00 | 1955.00 | 1963.20 | 1953.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 13:45:00 | 1953.25 | 1963.20 | 1953.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 14:15:00 | 1955.00 | 1961.56 | 1953.73 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-11-09 09:15:00 | 1938.30 | 1956.66 | 1952.85 | SL hit (close<ema400) qty=1.00 sl=1952.85 alert=retest1 |

### Cycle 32 — SELL (started 2023-11-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 10:15:00 | 1908.25 | 1946.97 | 1948.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 11:15:00 | 1898.00 | 1937.18 | 1944.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-12 18:15:00 | 1916.00 | 1862.68 | 1884.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-12 18:15:00 | 1916.00 | 1862.68 | 1884.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 1916.00 | 1862.68 | 1884.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-12 19:00:00 | 1916.00 | 1862.68 | 1884.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 09:15:00 | 1872.00 | 1864.54 | 1883.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-15 14:30:00 | 1856.10 | 1866.76 | 1873.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-15 15:15:00 | 1855.00 | 1866.76 | 1873.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-16 10:00:00 | 1857.50 | 1863.03 | 1870.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-16 10:45:00 | 1857.65 | 1861.56 | 1868.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 09:15:00 | 1851.80 | 1838.98 | 1848.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-20 09:30:00 | 1853.00 | 1838.98 | 1848.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 10:15:00 | 1840.00 | 1839.18 | 1847.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-20 10:30:00 | 1849.85 | 1839.18 | 1847.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 11:15:00 | 1849.15 | 1841.18 | 1847.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-20 12:00:00 | 1849.15 | 1841.18 | 1847.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 12:15:00 | 1843.80 | 1841.70 | 1847.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-20 14:00:00 | 1834.05 | 1840.17 | 1845.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-21 09:15:00 | 1852.75 | 1843.18 | 1845.93 | SL hit (close>static) qty=1.00 sl=1850.00 alert=retest2 |

### Cycle 33 — BUY (started 2023-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 13:15:00 | 1852.05 | 1847.99 | 1847.63 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-21 14:15:00 | 1841.25 | 1846.64 | 1847.05 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-22 09:15:00 | 1856.50 | 1848.35 | 1847.74 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 11:15:00 | 1838.60 | 1847.30 | 1847.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 13:15:00 | 1833.10 | 1843.27 | 1845.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-22 14:15:00 | 1843.35 | 1843.28 | 1845.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 14:15:00 | 1843.35 | 1843.28 | 1845.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 14:15:00 | 1843.35 | 1843.28 | 1845.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-22 14:30:00 | 1844.15 | 1843.28 | 1845.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 15:15:00 | 1849.20 | 1844.47 | 1845.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 09:15:00 | 1851.00 | 1844.47 | 1845.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 1851.80 | 1845.93 | 1846.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 09:30:00 | 1852.25 | 1845.93 | 1846.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 10:15:00 | 1840.00 | 1844.75 | 1845.65 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-11-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 12:15:00 | 1860.10 | 1847.55 | 1846.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-24 09:15:00 | 1937.75 | 1868.70 | 1857.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-28 12:15:00 | 1948.00 | 1948.31 | 1919.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-28 13:00:00 | 1948.00 | 1948.31 | 1919.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 10:15:00 | 2049.10 | 2062.46 | 2044.90 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2023-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-06 12:15:00 | 2016.00 | 2036.84 | 2039.51 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-12-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 09:15:00 | 2048.35 | 2041.22 | 2040.97 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2023-12-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-07 11:15:00 | 2025.50 | 2038.93 | 2040.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 09:15:00 | 2016.00 | 2026.60 | 2032.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-08 15:15:00 | 2012.00 | 2011.37 | 2021.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-11 09:15:00 | 2021.60 | 2011.37 | 2021.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 2060.00 | 2021.10 | 2025.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-11 09:30:00 | 2056.55 | 2021.10 | 2025.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 10:15:00 | 2053.95 | 2027.67 | 2027.68 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2023-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 11:15:00 | 2059.45 | 2034.03 | 2030.57 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2023-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 12:15:00 | 2018.00 | 2028.50 | 2029.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-13 10:15:00 | 2016.10 | 2024.67 | 2027.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 15:15:00 | 2023.50 | 2022.07 | 2024.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-14 09:15:00 | 2029.30 | 2022.07 | 2024.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 2018.00 | 2021.26 | 2024.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-14 10:30:00 | 2015.05 | 2021.27 | 2023.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-14 11:30:00 | 2012.60 | 2019.11 | 2022.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-15 11:00:00 | 2015.00 | 2009.59 | 2014.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-18 10:30:00 | 2014.85 | 2007.76 | 2010.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 13:15:00 | 2004.10 | 2005.30 | 2008.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-18 13:30:00 | 2007.25 | 2005.30 | 2008.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 14:15:00 | 2001.35 | 2004.51 | 2007.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-18 14:30:00 | 2005.00 | 2004.51 | 2007.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 2005.05 | 2003.74 | 2006.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-19 09:30:00 | 2011.95 | 2003.74 | 2006.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 10:15:00 | 2018.15 | 2006.62 | 2007.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-19 10:30:00 | 2032.70 | 2006.62 | 2007.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 11:15:00 | 2010.00 | 2007.30 | 2008.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-19 11:45:00 | 2012.05 | 2007.30 | 2008.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 12:15:00 | 2010.00 | 2007.84 | 2008.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-19 12:30:00 | 2012.00 | 2007.84 | 2008.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 13:15:00 | 1995.15 | 2005.30 | 2007.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-19 14:30:00 | 1990.10 | 2000.62 | 2004.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-20 11:45:00 | 1992.10 | 1997.32 | 2001.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-20 12:15:00 | 1992.60 | 1997.32 | 2001.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 14:15:00 | 1914.30 | 1971.89 | 1988.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 14:15:00 | 1911.97 | 1971.89 | 1988.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 14:15:00 | 1914.25 | 1971.89 | 1988.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 14:15:00 | 1914.11 | 1971.89 | 1988.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-12-22 09:15:00 | 1980.80 | 1954.75 | 1965.60 | SL hit (close>ema200) qty=0.50 sl=1954.75 alert=retest2 |

### Cycle 43 — BUY (started 2024-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-02 11:15:00 | 1896.85 | 1873.12 | 1872.43 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-03 13:15:00 | 1860.50 | 1874.92 | 1876.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-03 14:15:00 | 1855.95 | 1871.13 | 1874.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-04 09:15:00 | 1878.65 | 1871.65 | 1874.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-04 09:15:00 | 1878.65 | 1871.65 | 1874.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 09:15:00 | 1878.65 | 1871.65 | 1874.37 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2024-01-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-04 11:15:00 | 1885.05 | 1876.59 | 1876.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-05 09:15:00 | 1917.80 | 1886.57 | 1881.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-05 13:15:00 | 1882.90 | 1888.26 | 1883.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-05 13:15:00 | 1882.90 | 1888.26 | 1883.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 13:15:00 | 1882.90 | 1888.26 | 1883.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-05 14:00:00 | 1882.90 | 1888.26 | 1883.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 14:15:00 | 1905.00 | 1891.61 | 1885.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-08 10:15:00 | 1917.50 | 1895.48 | 1888.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-15 14:15:00 | 1958.00 | 1968.04 | 1969.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2024-01-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 14:15:00 | 1958.00 | 1968.04 | 1969.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 11:15:00 | 1953.60 | 1962.17 | 1965.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-17 09:15:00 | 1962.85 | 1947.13 | 1955.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-17 09:15:00 | 1962.85 | 1947.13 | 1955.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 1962.85 | 1947.13 | 1955.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-17 09:30:00 | 1963.00 | 1947.13 | 1955.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 10:15:00 | 1959.95 | 1949.69 | 1955.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-17 10:45:00 | 1961.30 | 1949.69 | 1955.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 12:15:00 | 1958.95 | 1952.95 | 1956.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-17 13:15:00 | 1970.00 | 1952.95 | 1956.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2024-01-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-17 14:15:00 | 1975.45 | 1961.37 | 1959.82 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-01-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 09:15:00 | 1928.45 | 1957.45 | 1958.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-20 09:15:00 | 1899.50 | 1924.56 | 1935.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 10:15:00 | 1870.00 | 1868.42 | 1887.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-24 11:00:00 | 1870.00 | 1868.42 | 1887.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 1901.00 | 1873.74 | 1881.12 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2024-01-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 12:15:00 | 1900.40 | 1886.86 | 1885.97 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-01-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 13:15:00 | 1870.00 | 1883.49 | 1884.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 14:15:00 | 1854.70 | 1877.73 | 1881.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-29 14:15:00 | 1871.10 | 1865.70 | 1872.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-29 15:00:00 | 1871.10 | 1865.70 | 1872.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 15:15:00 | 1873.55 | 1867.27 | 1872.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-30 09:15:00 | 1900.90 | 1867.27 | 1872.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2024-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 09:15:00 | 1937.85 | 1881.39 | 1878.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-30 10:15:00 | 1942.20 | 1893.55 | 1884.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-31 09:15:00 | 1906.55 | 1928.89 | 1910.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 09:15:00 | 1906.55 | 1928.89 | 1910.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 09:15:00 | 1906.55 | 1928.89 | 1910.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 10:00:00 | 1906.55 | 1928.89 | 1910.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 10:15:00 | 1890.30 | 1921.17 | 1908.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 10:45:00 | 1886.05 | 1921.17 | 1908.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 11:15:00 | 1909.05 | 1918.75 | 1908.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 09:15:00 | 1989.90 | 1910.08 | 1906.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-02 14:15:00 | 1912.10 | 1924.21 | 1924.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2024-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 14:15:00 | 1912.10 | 1924.21 | 1924.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 09:15:00 | 1895.10 | 1916.43 | 1920.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 13:15:00 | 1890.15 | 1884.75 | 1896.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-06 14:00:00 | 1890.15 | 1884.75 | 1896.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 09:15:00 | 1912.00 | 1890.94 | 1896.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 11:00:00 | 1897.25 | 1892.20 | 1896.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 11:45:00 | 1892.10 | 1892.27 | 1896.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 12:45:00 | 1893.50 | 1893.43 | 1896.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 13:15:00 | 1894.10 | 1893.43 | 1896.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 13:15:00 | 1897.55 | 1894.25 | 1896.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-07 14:00:00 | 1897.55 | 1894.25 | 1896.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 14:15:00 | 1891.70 | 1893.74 | 1895.98 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-02-08 11:15:00 | 1903.00 | 1897.85 | 1897.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2024-02-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-08 11:15:00 | 1903.00 | 1897.85 | 1897.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-08 12:15:00 | 1911.20 | 1900.52 | 1898.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-08 15:15:00 | 1900.05 | 1900.93 | 1899.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 15:15:00 | 1900.05 | 1900.93 | 1899.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 15:15:00 | 1900.05 | 1900.93 | 1899.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 09:15:00 | 1902.85 | 1900.93 | 1899.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 1888.50 | 1898.44 | 1898.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 09:45:00 | 1868.70 | 1898.44 | 1898.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2024-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 10:15:00 | 1865.95 | 1891.94 | 1895.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 15:15:00 | 1848.00 | 1859.34 | 1871.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-14 10:15:00 | 1812.30 | 1810.19 | 1831.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-14 10:45:00 | 1820.80 | 1810.19 | 1831.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 12:15:00 | 1870.00 | 1823.40 | 1834.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 13:00:00 | 1870.00 | 1823.40 | 1834.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 13:15:00 | 1877.20 | 1834.16 | 1838.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 14:00:00 | 1877.20 | 1834.16 | 1838.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2024-02-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 14:15:00 | 1873.60 | 1842.05 | 1841.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 09:15:00 | 1884.70 | 1855.20 | 1847.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-15 14:15:00 | 1854.70 | 1860.97 | 1854.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-15 14:15:00 | 1854.70 | 1860.97 | 1854.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 14:15:00 | 1854.70 | 1860.97 | 1854.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-15 14:30:00 | 1857.45 | 1860.97 | 1854.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 15:15:00 | 1859.00 | 1860.58 | 1854.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-16 09:15:00 | 1963.95 | 1860.58 | 1854.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-02-19 09:15:00 | 2160.35 | 2014.75 | 1951.39 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2024-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 09:15:00 | 2645.65 | 2724.87 | 2732.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 11:15:00 | 2589.65 | 2684.47 | 2711.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 15:15:00 | 2673.00 | 2651.34 | 2683.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-07 09:15:00 | 2715.05 | 2651.34 | 2683.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 2698.45 | 2660.76 | 2685.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 09:15:00 | 2640.00 | 2684.54 | 2688.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 09:15:00 | 2508.00 | 2576.92 | 2622.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-03-13 09:15:00 | 2376.00 | 2424.59 | 2512.55 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 57 — BUY (started 2024-03-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 11:15:00 | 2338.25 | 2295.96 | 2293.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-26 13:15:00 | 2376.95 | 2342.49 | 2329.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 14:15:00 | 2342.20 | 2342.43 | 2330.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-26 15:00:00 | 2342.20 | 2342.43 | 2330.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 14:15:00 | 2412.00 | 2435.52 | 2410.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 15:00:00 | 2412.00 | 2435.52 | 2410.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 15:15:00 | 2429.00 | 2434.21 | 2411.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-01 09:15:00 | 2503.65 | 2434.21 | 2411.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-02 09:15:00 | 2754.02 | 2559.92 | 2496.12 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2024-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 13:15:00 | 2642.40 | 2672.85 | 2676.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 13:15:00 | 2636.45 | 2661.94 | 2668.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-09 14:15:00 | 2685.90 | 2666.73 | 2670.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 14:15:00 | 2685.90 | 2666.73 | 2670.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 14:15:00 | 2685.90 | 2666.73 | 2670.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-09 15:00:00 | 2685.90 | 2666.73 | 2670.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 15:15:00 | 2693.00 | 2671.99 | 2672.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 09:15:00 | 2666.00 | 2671.99 | 2672.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2024-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 09:15:00 | 2737.35 | 2685.06 | 2678.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 10:15:00 | 2764.95 | 2701.04 | 2685.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-15 09:15:00 | 2899.50 | 2926.56 | 2867.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-15 09:15:00 | 2899.50 | 2926.56 | 2867.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 2899.50 | 2926.56 | 2867.93 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2024-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-16 11:15:00 | 2820.00 | 2849.84 | 2853.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-16 14:15:00 | 2807.05 | 2832.52 | 2844.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-19 09:15:00 | 2794.10 | 2776.04 | 2801.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-19 09:15:00 | 2794.10 | 2776.04 | 2801.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 2794.10 | 2776.04 | 2801.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 09:30:00 | 2871.95 | 2776.04 | 2801.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 10:15:00 | 2812.00 | 2783.23 | 2802.25 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2024-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 09:15:00 | 2914.80 | 2817.63 | 2811.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 09:15:00 | 3182.60 | 2942.76 | 2896.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 13:15:00 | 3066.00 | 3070.13 | 3019.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-25 13:45:00 | 3068.70 | 3070.13 | 3019.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 09:15:00 | 2959.00 | 3055.87 | 3045.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 09:45:00 | 2933.75 | 3055.87 | 3045.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 10:15:00 | 2989.50 | 3042.60 | 3040.72 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2024-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 11:15:00 | 2990.05 | 3032.09 | 3036.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-30 09:15:00 | 2968.95 | 3009.79 | 3023.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-30 13:15:00 | 3005.55 | 2999.34 | 3012.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-30 13:15:00 | 3005.55 | 2999.34 | 3012.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 13:15:00 | 3005.55 | 2999.34 | 3012.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 13:30:00 | 3013.55 | 2999.34 | 3012.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 14:15:00 | 2979.95 | 2995.46 | 3009.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-30 15:15:00 | 2962.00 | 2995.46 | 3009.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 09:45:00 | 2963.40 | 2983.78 | 3001.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 10:45:00 | 2970.00 | 2980.63 | 2998.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 09:15:00 | 2813.90 | 2890.82 | 2928.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 09:15:00 | 2815.23 | 2890.82 | 2928.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 09:15:00 | 2821.50 | 2890.82 | 2928.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-08 09:15:00 | 2828.00 | 2783.44 | 2824.54 | SL hit (close>ema200) qty=0.50 sl=2783.44 alert=retest2 |

### Cycle 63 — BUY (started 2024-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-09 10:15:00 | 2883.00 | 2839.40 | 2835.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-09 13:15:00 | 2923.05 | 2879.31 | 2856.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 14:15:00 | 2844.25 | 2872.30 | 2855.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-09 14:15:00 | 2844.25 | 2872.30 | 2855.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 14:15:00 | 2844.25 | 2872.30 | 2855.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 15:00:00 | 2844.25 | 2872.30 | 2855.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 15:15:00 | 2839.95 | 2865.83 | 2854.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-10 09:15:00 | 2827.90 | 2865.83 | 2854.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 2851.80 | 2863.02 | 2853.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-10 09:30:00 | 2823.25 | 2863.02 | 2853.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 10:15:00 | 2842.50 | 2858.92 | 2852.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-10 10:30:00 | 2847.55 | 2858.92 | 2852.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 11:15:00 | 2838.00 | 2854.73 | 2851.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-10 12:15:00 | 2813.00 | 2854.73 | 2851.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2024-05-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-10 12:15:00 | 2807.00 | 2845.19 | 2847.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-13 09:15:00 | 2783.00 | 2820.36 | 2834.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-14 09:15:00 | 2886.55 | 2806.17 | 2815.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-14 09:15:00 | 2886.55 | 2806.17 | 2815.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 2886.55 | 2806.17 | 2815.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:45:00 | 2907.40 | 2806.17 | 2815.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 2888.80 | 2822.69 | 2821.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 12:15:00 | 2958.55 | 2863.07 | 2841.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 11:15:00 | 2917.60 | 2918.95 | 2884.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 12:00:00 | 2917.60 | 2918.95 | 2884.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 14:15:00 | 2865.95 | 2902.03 | 2884.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 15:00:00 | 2865.95 | 2902.03 | 2884.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 2869.00 | 2895.42 | 2883.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 09:15:00 | 2893.65 | 2895.42 | 2883.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-17 13:15:00 | 3183.02 | 3070.35 | 2992.04 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2024-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 14:15:00 | 3044.00 | 3111.40 | 3118.67 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 11:15:00 | 3145.80 | 3119.97 | 3119.76 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2024-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 12:15:00 | 3110.10 | 3118.00 | 3118.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 13:15:00 | 3101.10 | 3114.62 | 3117.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 10:15:00 | 3110.00 | 3105.18 | 3111.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 10:15:00 | 3110.00 | 3105.18 | 3111.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 3110.00 | 3105.18 | 3111.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:45:00 | 3126.95 | 3105.18 | 3111.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 3110.00 | 3106.14 | 3111.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 11:45:00 | 3115.35 | 3106.14 | 3111.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 12:15:00 | 3072.25 | 3099.36 | 3107.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 13:15:00 | 3064.00 | 3099.36 | 3107.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 12:15:00 | 2910.80 | 2983.96 | 3038.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-29 09:15:00 | 2919.90 | 2879.45 | 2931.07 | SL hit (close>ema200) qty=0.50 sl=2879.45 alert=retest2 |

### Cycle 69 — BUY (started 2024-05-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 13:15:00 | 2989.40 | 2896.86 | 2885.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 14:15:00 | 2999.15 | 2917.32 | 2896.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-31 15:15:00 | 2908.05 | 2915.46 | 2897.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 15:15:00 | 2908.05 | 2915.46 | 2897.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 2908.05 | 2915.46 | 2897.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 09:15:00 | 3005.00 | 2915.46 | 2897.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 09:15:00 | 2824.95 | 2937.27 | 2927.40 | SL hit (close<static) qty=1.00 sl=2885.00 alert=retest2 |

### Cycle 70 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 2670.90 | 2883.99 | 2904.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 2580.00 | 2823.19 | 2874.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 2691.45 | 2569.14 | 2654.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 2691.45 | 2569.14 | 2654.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 2691.45 | 2569.14 | 2654.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 2691.45 | 2569.14 | 2654.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 2659.65 | 2587.24 | 2655.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 12:00:00 | 2627.05 | 2595.20 | 2652.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 13:45:00 | 2631.90 | 2609.75 | 2649.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-07 10:00:00 | 2628.55 | 2617.76 | 2643.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-07 10:45:00 | 2633.35 | 2622.91 | 2643.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 11:15:00 | 2639.50 | 2626.23 | 2643.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 11:30:00 | 2642.35 | 2626.23 | 2643.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 12:15:00 | 2650.80 | 2631.14 | 2644.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 12:30:00 | 2664.80 | 2631.14 | 2644.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 13:15:00 | 2635.00 | 2631.91 | 2643.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 13:30:00 | 2641.95 | 2631.91 | 2643.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 09:15:00 | 2646.00 | 2636.07 | 2642.47 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-06-10 13:15:00 | 2670.00 | 2649.54 | 2647.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2024-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-10 13:15:00 | 2670.00 | 2649.54 | 2647.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-11 09:15:00 | 2712.10 | 2662.19 | 2653.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 14:15:00 | 2671.05 | 2685.73 | 2670.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 15:00:00 | 2671.05 | 2685.73 | 2670.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 2668.50 | 2682.28 | 2670.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 2724.55 | 2682.28 | 2670.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-18 09:15:00 | 2997.01 | 2894.54 | 2820.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2024-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 12:15:00 | 2943.00 | 2965.50 | 2968.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 15:15:00 | 2930.00 | 2957.95 | 2964.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 09:15:00 | 2989.50 | 2964.26 | 2966.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 09:15:00 | 2989.50 | 2964.26 | 2966.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 2989.50 | 2964.26 | 2966.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:00:00 | 2989.50 | 2964.26 | 2966.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 2964.55 | 2964.32 | 2966.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 11:30:00 | 2940.00 | 2964.67 | 2966.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 13:15:00 | 2992.00 | 2970.99 | 2968.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2024-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 13:15:00 | 2992.00 | 2970.99 | 2968.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 09:15:00 | 3046.30 | 2992.41 | 2979.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 14:15:00 | 2980.50 | 2998.59 | 2989.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 14:15:00 | 2980.50 | 2998.59 | 2989.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 2980.50 | 2998.59 | 2989.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 15:00:00 | 2980.50 | 2998.59 | 2989.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 15:15:00 | 2967.00 | 2992.27 | 2987.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 09:15:00 | 3009.55 | 2992.27 | 2987.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 13:30:00 | 2990.95 | 2990.11 | 2988.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 15:15:00 | 2991.00 | 2987.90 | 2987.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-05 09:15:00 | 3290.05 | 3187.11 | 3139.59 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2024-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 12:15:00 | 3224.40 | 3282.86 | 3288.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 10:15:00 | 3193.65 | 3250.15 | 3270.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 09:15:00 | 3239.00 | 3223.99 | 3244.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 09:15:00 | 3239.00 | 3223.99 | 3244.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 3239.00 | 3223.99 | 3244.90 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2024-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 12:15:00 | 3377.05 | 3255.36 | 3254.08 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-07-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 12:15:00 | 3314.05 | 3328.56 | 3329.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 13:15:00 | 3278.65 | 3318.58 | 3324.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-19 11:15:00 | 3220.00 | 3202.27 | 3240.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-19 12:00:00 | 3220.00 | 3202.27 | 3240.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 3209.00 | 3196.77 | 3220.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:00:00 | 3209.00 | 3196.77 | 3220.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 3204.70 | 3198.35 | 3219.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 14:45:00 | 3190.00 | 3202.58 | 3216.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 09:15:00 | 3182.35 | 3201.66 | 3214.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 11:15:00 | 3175.40 | 3200.39 | 3211.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 3030.50 | 3164.74 | 3193.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 3023.23 | 3164.74 | 3193.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 3016.63 | 3164.74 | 3193.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-24 15:15:00 | 3111.00 | 3099.06 | 3130.99 | SL hit (close>ema200) qty=0.50 sl=3099.06 alert=retest2 |

### Cycle 77 — BUY (started 2024-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 10:15:00 | 3414.15 | 3201.17 | 3174.28 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2024-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 14:15:00 | 3280.00 | 3303.03 | 3305.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 09:15:00 | 3241.55 | 3287.53 | 3297.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 10:15:00 | 3184.95 | 3178.45 | 3212.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-02 11:00:00 | 3184.95 | 3178.45 | 3212.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 3015.50 | 3010.67 | 3078.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 10:30:00 | 2978.10 | 3002.87 | 3068.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 11:00:00 | 2971.70 | 3002.87 | 3068.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 10:00:00 | 2989.55 | 2958.04 | 2976.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 13:15:00 | 2978.65 | 2981.29 | 2983.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 2969.20 | 2971.52 | 2977.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 10:15:00 | 2953.10 | 2971.52 | 2977.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 13:30:00 | 2962.85 | 2968.81 | 2974.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 14:30:00 | 2958.10 | 2968.05 | 2973.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 15:15:00 | 2956.00 | 2968.05 | 2973.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 2956.00 | 2965.64 | 2971.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:15:00 | 2916.45 | 2965.64 | 2971.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 11:15:00 | 3008.90 | 2938.82 | 2945.16 | SL hit (close>static) qty=1.00 sl=2975.45 alert=retest2 |

### Cycle 79 — BUY (started 2024-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 12:15:00 | 3042.00 | 2959.46 | 2953.96 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2024-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 11:15:00 | 2918.00 | 2956.02 | 2958.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-19 09:15:00 | 2867.40 | 2920.04 | 2935.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 09:15:00 | 2890.40 | 2867.02 | 2882.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 09:15:00 | 2890.40 | 2867.02 | 2882.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 2890.40 | 2867.02 | 2882.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:45:00 | 2902.00 | 2867.02 | 2882.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 10:15:00 | 2875.90 | 2868.80 | 2881.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 10:45:00 | 2892.50 | 2868.80 | 2881.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 11:15:00 | 2880.00 | 2871.04 | 2881.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 12:00:00 | 2880.00 | 2871.04 | 2881.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 12:15:00 | 2872.85 | 2871.40 | 2880.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 13:15:00 | 2907.50 | 2871.40 | 2880.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 13:15:00 | 2877.45 | 2872.61 | 2880.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-21 14:15:00 | 2870.35 | 2872.61 | 2880.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-22 13:00:00 | 2870.75 | 2876.69 | 2879.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-22 14:30:00 | 2870.00 | 2875.29 | 2878.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-22 15:15:00 | 2869.00 | 2875.29 | 2878.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 2882.30 | 2875.69 | 2878.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:00:00 | 2882.30 | 2875.69 | 2878.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 2882.30 | 2877.01 | 2878.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-23 11:30:00 | 2879.00 | 2876.81 | 2878.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-23 14:15:00 | 2877.40 | 2878.55 | 2878.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 14:15:00 | 2885.05 | 2879.85 | 2879.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2024-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 14:15:00 | 2885.05 | 2879.85 | 2879.35 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 13:15:00 | 2863.25 | 2876.76 | 2878.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 15:15:00 | 2860.00 | 2871.46 | 2875.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 09:15:00 | 2880.00 | 2873.17 | 2875.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 2880.00 | 2873.17 | 2875.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 2880.00 | 2873.17 | 2875.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 12:45:00 | 2862.45 | 2874.20 | 2875.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 09:15:00 | 2852.10 | 2871.13 | 2873.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-05 10:15:00 | 2793.10 | 2777.21 | 2776.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2024-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 10:15:00 | 2793.10 | 2777.21 | 2776.48 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 09:15:00 | 2743.45 | 2778.62 | 2778.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 2736.00 | 2754.06 | 2765.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 14:15:00 | 2712.80 | 2702.67 | 2728.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 15:00:00 | 2712.80 | 2702.67 | 2728.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 2765.05 | 2715.15 | 2731.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:15:00 | 2763.75 | 2715.15 | 2731.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 2724.00 | 2716.92 | 2730.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 09:15:00 | 2708.15 | 2727.51 | 2730.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-16 14:15:00 | 2749.95 | 2701.89 | 2696.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2024-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 14:15:00 | 2749.95 | 2701.89 | 2696.36 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 11:15:00 | 2687.85 | 2701.29 | 2701.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 12:15:00 | 2670.00 | 2695.03 | 2698.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 10:15:00 | 2571.35 | 2561.79 | 2605.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 10:45:00 | 2562.05 | 2561.79 | 2605.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 2577.10 | 2555.12 | 2582.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:30:00 | 2618.75 | 2555.12 | 2582.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 2556.65 | 2555.42 | 2579.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 11:45:00 | 2549.55 | 2554.28 | 2577.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 13:30:00 | 2548.00 | 2549.17 | 2570.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 09:15:00 | 2508.40 | 2553.05 | 2568.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 09:15:00 | 2422.07 | 2452.57 | 2481.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 09:15:00 | 2420.60 | 2452.57 | 2481.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 14:15:00 | 2382.98 | 2422.02 | 2454.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-01 10:15:00 | 2344.60 | 2342.27 | 2379.53 | SL hit (close>ema200) qty=0.50 sl=2342.27 alert=retest2 |

### Cycle 87 — BUY (started 2024-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 13:15:00 | 2348.65 | 2301.28 | 2298.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 14:15:00 | 2375.15 | 2316.05 | 2305.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-09 14:15:00 | 2389.55 | 2392.98 | 2358.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-09 15:00:00 | 2389.55 | 2392.98 | 2358.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 12:15:00 | 2444.95 | 2465.30 | 2452.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 12:45:00 | 2447.05 | 2465.30 | 2452.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 13:15:00 | 2441.35 | 2460.51 | 2451.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 14:00:00 | 2441.35 | 2460.51 | 2451.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 14:15:00 | 2442.80 | 2456.97 | 2450.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 14:30:00 | 2439.55 | 2456.97 | 2450.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 2460.00 | 2504.52 | 2492.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:45:00 | 2454.00 | 2504.52 | 2492.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 2453.00 | 2494.21 | 2489.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:00:00 | 2453.00 | 2494.21 | 2489.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2024-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 12:15:00 | 2452.05 | 2479.34 | 2482.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 2405.10 | 2453.22 | 2468.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 11:15:00 | 2467.85 | 2453.86 | 2466.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 11:15:00 | 2467.85 | 2453.86 | 2466.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 2467.85 | 2453.86 | 2466.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:00:00 | 2467.85 | 2453.86 | 2466.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 2456.70 | 2454.43 | 2465.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 13:15:00 | 2466.20 | 2454.43 | 2465.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 2467.85 | 2457.11 | 2465.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 15:00:00 | 2451.70 | 2456.03 | 2464.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 09:15:00 | 2425.30 | 2456.43 | 2463.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 09:15:00 | 2329.11 | 2383.10 | 2416.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:15:00 | 2304.03 | 2369.38 | 2406.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-23 09:15:00 | 2206.53 | 2299.41 | 2350.83 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 89 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 2341.00 | 2231.28 | 2217.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 11:15:00 | 2428.00 | 2288.12 | 2246.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 11:15:00 | 2398.00 | 2398.23 | 2337.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 11:30:00 | 2397.20 | 2398.23 | 2337.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 2351.95 | 2404.79 | 2372.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 2351.95 | 2404.79 | 2372.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 2369.55 | 2397.75 | 2372.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 11:45:00 | 2382.05 | 2392.22 | 2372.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 12:30:00 | 2373.95 | 2389.03 | 2372.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 13:00:00 | 2376.30 | 2389.03 | 2372.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 10:15:00 | 2341.00 | 2366.46 | 2366.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2024-11-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 10:15:00 | 2341.00 | 2366.46 | 2366.59 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 2403.30 | 2367.57 | 2365.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 09:15:00 | 2455.90 | 2412.42 | 2392.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 11:15:00 | 2415.25 | 2417.10 | 2398.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 11:45:00 | 2416.60 | 2417.10 | 2398.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 2390.05 | 2411.95 | 2400.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:00:00 | 2390.05 | 2411.95 | 2400.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 2395.00 | 2408.56 | 2400.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 2376.70 | 2408.56 | 2400.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 2369.20 | 2400.69 | 2397.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:30:00 | 2373.40 | 2400.69 | 2397.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 2346.05 | 2389.76 | 2392.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 2329.30 | 2377.67 | 2387.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 2233.90 | 2229.94 | 2282.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 2233.90 | 2229.94 | 2282.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 2233.90 | 2229.94 | 2282.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 10:30:00 | 2222.95 | 2229.83 | 2277.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 09:15:00 | 2206.20 | 2240.50 | 2265.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 10:15:00 | 2300.00 | 2209.76 | 2200.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 2300.00 | 2209.76 | 2200.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 09:15:00 | 2395.00 | 2323.76 | 2300.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 12:15:00 | 2525.30 | 2529.22 | 2505.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-03 12:45:00 | 2521.90 | 2529.22 | 2505.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 15:15:00 | 2515.60 | 2524.67 | 2509.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 09:15:00 | 2598.70 | 2524.67 | 2509.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 12:15:00 | 2611.75 | 2633.11 | 2634.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2024-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 12:15:00 | 2611.75 | 2633.11 | 2634.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 14:15:00 | 2603.35 | 2624.07 | 2630.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 09:15:00 | 2618.05 | 2567.56 | 2586.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 09:15:00 | 2618.05 | 2567.56 | 2586.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 2618.05 | 2567.56 | 2586.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:00:00 | 2618.05 | 2567.56 | 2586.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 2608.60 | 2575.76 | 2588.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:30:00 | 2617.20 | 2575.76 | 2588.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2024-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 12:15:00 | 2650.00 | 2599.39 | 2597.58 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 09:15:00 | 2561.35 | 2605.28 | 2608.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 11:15:00 | 2553.30 | 2588.75 | 2599.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 09:15:00 | 2537.65 | 2513.60 | 2538.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 09:15:00 | 2537.65 | 2513.60 | 2538.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 2537.65 | 2513.60 | 2538.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 10:00:00 | 2537.65 | 2513.60 | 2538.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 2546.00 | 2520.08 | 2539.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 11:15:00 | 2592.65 | 2520.08 | 2539.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 2561.75 | 2528.42 | 2541.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:30:00 | 2529.80 | 2531.42 | 2541.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 13:30:00 | 2531.90 | 2524.17 | 2537.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 14:45:00 | 2522.25 | 2532.88 | 2540.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 09:15:00 | 2468.50 | 2536.51 | 2541.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 2525.30 | 2498.60 | 2512.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 14:15:00 | 2492.00 | 2504.12 | 2511.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 14:45:00 | 2495.00 | 2500.61 | 2508.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 11:30:00 | 2493.65 | 2498.56 | 2505.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 09:30:00 | 2492.85 | 2489.86 | 2497.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 2504.50 | 2491.34 | 2496.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 13:00:00 | 2504.50 | 2491.34 | 2496.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-27 13:15:00 | 2535.15 | 2500.10 | 2499.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2024-12-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 13:15:00 | 2535.15 | 2500.10 | 2499.78 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 12:15:00 | 2477.95 | 2498.76 | 2500.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 13:15:00 | 2462.70 | 2491.54 | 2496.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 13:15:00 | 2470.35 | 2452.60 | 2468.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 13:15:00 | 2470.35 | 2452.60 | 2468.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 2470.35 | 2452.60 | 2468.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 13:45:00 | 2467.30 | 2452.60 | 2468.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 2469.45 | 2455.97 | 2468.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:30:00 | 2470.80 | 2455.97 | 2468.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 15:15:00 | 2465.10 | 2457.80 | 2468.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:15:00 | 2489.00 | 2457.80 | 2468.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 2487.25 | 2463.69 | 2469.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:45:00 | 2498.05 | 2463.69 | 2469.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 2491.00 | 2469.15 | 2471.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 11:00:00 | 2491.00 | 2469.15 | 2471.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 12:15:00 | 2486.05 | 2474.87 | 2474.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 13:15:00 | 2500.45 | 2479.99 | 2476.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 11:15:00 | 2486.20 | 2488.65 | 2483.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 11:15:00 | 2486.20 | 2488.65 | 2483.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 2486.20 | 2488.65 | 2483.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 14:45:00 | 2494.50 | 2488.56 | 2484.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 09:15:00 | 2520.00 | 2488.45 | 2484.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 09:15:00 | 2444.75 | 2487.85 | 2489.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 2444.75 | 2487.85 | 2489.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 2388.85 | 2468.05 | 2480.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 13:15:00 | 2390.00 | 2387.15 | 2416.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 13:30:00 | 2389.65 | 2387.15 | 2416.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 2384.70 | 2388.10 | 2409.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 11:00:00 | 2352.50 | 2380.98 | 2404.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 11:30:00 | 2352.40 | 2370.90 | 2397.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 2234.88 | 2307.18 | 2338.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 2234.78 | 2307.18 | 2338.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-10 10:15:00 | 2316.95 | 2309.13 | 2336.20 | SL hit (close>ema200) qty=0.50 sl=2309.13 alert=retest2 |

### Cycle 101 — BUY (started 2025-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 12:15:00 | 2197.90 | 2178.86 | 2177.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 2239.10 | 2197.59 | 2187.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 2316.80 | 2326.87 | 2286.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:00:00 | 2316.80 | 2326.87 | 2286.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 2302.50 | 2315.48 | 2293.19 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 2198.50 | 2270.20 | 2277.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 2156.90 | 2247.54 | 2266.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 2239.40 | 2219.90 | 2242.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 2239.40 | 2219.90 | 2242.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 2239.40 | 2219.90 | 2242.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 2237.10 | 2219.90 | 2242.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 2241.10 | 2224.14 | 2242.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 2251.70 | 2224.14 | 2242.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 2199.90 | 2219.29 | 2238.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 14:00:00 | 2189.20 | 2210.03 | 2230.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 2079.74 | 2119.81 | 2164.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-28 09:15:00 | 1970.28 | 1998.74 | 2071.70 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 103 — BUY (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 12:15:00 | 2092.90 | 2061.54 | 2060.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 2103.00 | 2073.66 | 2066.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 15:15:00 | 2112.20 | 2123.80 | 2103.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 09:15:00 | 2129.00 | 2123.80 | 2103.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 2170.50 | 2133.14 | 2109.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 10:15:00 | 2185.00 | 2133.14 | 2109.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 12:15:00 | 2072.55 | 2191.63 | 2168.08 | SL hit (close<static) qty=1.00 sl=2109.10 alert=retest2 |

### Cycle 104 — SELL (started 2025-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 15:15:00 | 2105.00 | 2147.59 | 2151.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 10:15:00 | 2059.15 | 2119.74 | 2137.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 13:15:00 | 2022.05 | 2012.38 | 2053.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 14:00:00 | 2022.05 | 2012.38 | 2053.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 2059.90 | 2024.41 | 2048.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 09:15:00 | 1973.75 | 2030.89 | 2042.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 10:15:00 | 1875.06 | 1915.13 | 1951.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-11 11:15:00 | 1776.38 | 1836.03 | 1887.26 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 105 — BUY (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 10:15:00 | 1592.80 | 1547.06 | 1546.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-24 11:15:00 | 1656.20 | 1582.50 | 1569.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-25 11:15:00 | 1608.75 | 1616.08 | 1597.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-25 11:45:00 | 1605.50 | 1616.08 | 1597.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 15:15:00 | 1603.70 | 1611.47 | 1601.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 09:15:00 | 1589.15 | 1611.47 | 1601.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 1579.30 | 1605.03 | 1599.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 10:00:00 | 1579.30 | 1605.03 | 1599.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 10:15:00 | 1559.00 | 1595.83 | 1595.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 11:00:00 | 1559.00 | 1595.83 | 1595.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2025-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 11:15:00 | 1558.45 | 1588.35 | 1592.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 12:15:00 | 1553.70 | 1581.42 | 1588.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 13:15:00 | 1423.70 | 1415.33 | 1462.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 14:00:00 | 1423.70 | 1415.33 | 1462.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 1441.85 | 1424.82 | 1455.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:45:00 | 1451.00 | 1424.82 | 1455.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 1411.05 | 1422.07 | 1451.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 11:30:00 | 1404.20 | 1416.84 | 1446.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 10:00:00 | 1400.75 | 1409.11 | 1430.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 14:30:00 | 1406.80 | 1413.65 | 1425.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-07 09:15:00 | 1521.30 | 1438.79 | 1430.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2025-03-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-07 09:15:00 | 1521.30 | 1438.79 | 1430.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 10:15:00 | 1560.10 | 1463.06 | 1442.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 15:15:00 | 1603.00 | 1613.35 | 1563.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-11 09:15:00 | 1582.75 | 1613.35 | 1563.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 1612.65 | 1613.21 | 1567.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:30:00 | 1570.05 | 1613.21 | 1567.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 1580.00 | 1602.38 | 1570.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 09:15:00 | 1605.20 | 1587.53 | 1572.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 13:00:00 | 1594.00 | 1587.48 | 1576.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 09:45:00 | 1666.45 | 1604.79 | 1587.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-19 11:15:00 | 1753.40 | 1671.49 | 1654.49 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 11:15:00 | 1715.00 | 1720.07 | 1720.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 15:15:00 | 1687.00 | 1708.16 | 1714.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 12:15:00 | 1706.45 | 1702.43 | 1708.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 12:15:00 | 1706.45 | 1702.43 | 1708.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 1706.45 | 1702.43 | 1708.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:00:00 | 1706.45 | 1702.43 | 1708.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 1699.00 | 1701.75 | 1708.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:30:00 | 1703.70 | 1701.75 | 1708.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 1720.15 | 1702.43 | 1706.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:30:00 | 1727.15 | 1702.43 | 1706.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 1708.80 | 1703.71 | 1706.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 11:15:00 | 1707.35 | 1703.71 | 1706.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 14:15:00 | 1700.50 | 1695.25 | 1698.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 13:15:00 | 1705.00 | 1698.67 | 1698.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 13:15:00 | 1705.00 | 1698.67 | 1698.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 14:15:00 | 1722.40 | 1703.41 | 1700.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 1679.00 | 1725.39 | 1719.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 1679.00 | 1725.39 | 1719.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1679.00 | 1725.39 | 1719.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 1679.00 | 1725.39 | 1719.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 1686.00 | 1717.51 | 1716.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:30:00 | 1677.65 | 1717.51 | 1716.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 1671.05 | 1708.22 | 1712.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 1662.80 | 1699.14 | 1707.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-04 15:15:00 | 1691.95 | 1689.60 | 1700.49 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-07 09:15:00 | 1597.50 | 1689.60 | 1700.49 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-07 09:15:00 | 1437.75 | 1663.17 | 1687.49 | Target hit (10%) qty=1.00 alert=retest1 |

### Cycle 111 — BUY (started 2025-04-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 12:15:00 | 1665.55 | 1651.21 | 1649.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 14:15:00 | 1683.25 | 1659.03 | 1653.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 1860.60 | 1871.84 | 1824.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:00:00 | 1860.60 | 1871.84 | 1824.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 2095.60 | 2138.04 | 2105.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 2095.60 | 2138.04 | 2105.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 2059.30 | 2122.30 | 2100.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 2059.30 | 2122.30 | 2100.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 11:15:00 | 2068.00 | 2111.44 | 2097.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:45:00 | 2041.10 | 2111.44 | 2097.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 14:15:00 | 2092.70 | 2102.67 | 2096.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 15:00:00 | 2092.70 | 2102.67 | 2096.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 15:15:00 | 2072.00 | 2096.54 | 2094.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-28 09:15:00 | 2109.70 | 2096.54 | 2094.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 2237.60 | 2124.75 | 2107.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-28 10:30:00 | 2268.00 | 2154.96 | 2122.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-28 14:45:00 | 2247.00 | 2211.68 | 2163.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-29 09:15:00 | 2307.70 | 2213.34 | 2169.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-29 12:15:00 | 2494.80 | 2328.65 | 2242.83 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2025-05-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 14:15:00 | 2316.30 | 2380.31 | 2380.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-05 09:15:00 | 2219.80 | 2337.96 | 2360.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 2237.80 | 2234.98 | 2269.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 10:30:00 | 2228.60 | 2234.98 | 2269.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 2260.00 | 2207.87 | 2236.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 10:15:00 | 2256.70 | 2207.87 | 2236.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 2239.20 | 2214.14 | 2236.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:30:00 | 2219.20 | 2221.90 | 2235.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 11:15:00 | 2284.20 | 2242.11 | 2239.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2025-05-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 11:15:00 | 2284.20 | 2242.11 | 2239.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-09 12:15:00 | 2289.00 | 2251.49 | 2244.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 2832.60 | 2866.19 | 2772.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 14:00:00 | 2832.60 | 2866.19 | 2772.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 2724.70 | 2820.63 | 2773.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:15:00 | 2696.60 | 2820.63 | 2773.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 2733.50 | 2803.20 | 2769.77 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 2682.00 | 2750.94 | 2751.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 2626.60 | 2726.07 | 2740.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 2781.50 | 2726.41 | 2737.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 2781.50 | 2726.41 | 2737.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 2781.50 | 2726.41 | 2737.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:45:00 | 2756.80 | 2726.41 | 2737.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 2737.60 | 2728.65 | 2737.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:45:00 | 2780.00 | 2728.65 | 2737.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 2746.00 | 2732.12 | 2738.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:45:00 | 2743.50 | 2732.12 | 2738.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 2743.00 | 2734.29 | 2738.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:30:00 | 2775.80 | 2734.29 | 2738.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 13:15:00 | 2741.00 | 2735.64 | 2738.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 14:15:00 | 2746.50 | 2735.64 | 2738.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 2745.00 | 2737.51 | 2739.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 15:00:00 | 2745.00 | 2737.51 | 2739.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 15:15:00 | 2754.00 | 2740.81 | 2740.74 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 09:15:00 | 2716.00 | 2735.85 | 2738.49 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 2799.90 | 2737.86 | 2732.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 11:15:00 | 2846.20 | 2759.53 | 2742.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 11:15:00 | 2813.80 | 2817.37 | 2787.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 11:45:00 | 2808.80 | 2817.37 | 2787.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 2889.40 | 2862.07 | 2842.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 09:45:00 | 2930.20 | 2879.57 | 2861.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 10:45:00 | 2923.00 | 2893.76 | 2869.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-06 09:15:00 | 3223.22 | 3117.99 | 3054.18 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 10:15:00 | 2965.60 | 3047.97 | 3049.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-09 14:15:00 | 2955.40 | 2996.85 | 3021.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 09:15:00 | 3086.00 | 3007.97 | 3021.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 09:15:00 | 3086.00 | 3007.97 | 3021.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 3086.00 | 3007.97 | 3021.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:00:00 | 3086.00 | 3007.97 | 3021.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 3094.70 | 3025.31 | 3028.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:00:00 | 3094.70 | 3025.31 | 3028.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2025-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 11:15:00 | 3091.40 | 3038.53 | 3034.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 13:15:00 | 3115.00 | 3064.65 | 3047.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 3048.80 | 3078.90 | 3059.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 3048.80 | 3078.90 | 3059.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 3048.80 | 3078.90 | 3059.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:45:00 | 3034.60 | 3078.90 | 3059.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 3061.70 | 3075.46 | 3060.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:45:00 | 3049.50 | 3075.46 | 3060.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 3055.30 | 3070.91 | 3060.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:00:00 | 3055.30 | 3070.91 | 3060.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 3008.00 | 3058.33 | 3055.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 2996.00 | 3058.33 | 3055.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 14:15:00 | 3034.50 | 3053.56 | 3053.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 2979.80 | 3034.08 | 3044.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 09:15:00 | 2986.00 | 2972.87 | 3001.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 10:00:00 | 2986.00 | 2972.87 | 3001.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 3000.00 | 2978.29 | 3001.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 10:45:00 | 3014.00 | 2978.29 | 3001.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 11:15:00 | 2985.00 | 2979.63 | 3000.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:30:00 | 2930.50 | 2984.75 | 2996.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 12:30:00 | 2971.80 | 2980.10 | 2991.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 09:15:00 | 3023.90 | 2982.73 | 2988.07 | SL hit (close>static) qty=1.00 sl=3014.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 3035.90 | 2993.36 | 2992.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-19 09:15:00 | 3081.80 | 3028.16 | 3015.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 10:15:00 | 3016.00 | 3025.73 | 3015.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 10:15:00 | 3016.00 | 3025.73 | 3015.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 3016.00 | 3025.73 | 3015.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:45:00 | 3026.10 | 3025.73 | 3015.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 2971.00 | 3014.78 | 3011.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:00:00 | 2971.00 | 3014.78 | 3011.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 2920.20 | 2995.87 | 3003.47 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 3028.00 | 2985.57 | 2984.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 3046.80 | 2997.82 | 2989.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 09:15:00 | 2946.40 | 3002.43 | 2997.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 2946.40 | 3002.43 | 2997.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 2946.40 | 3002.43 | 2997.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:45:00 | 2942.30 | 3002.43 | 2997.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 10:15:00 | 2939.20 | 2989.78 | 2991.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-24 11:15:00 | 2928.00 | 2977.43 | 2986.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 10:15:00 | 2864.30 | 2838.11 | 2882.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-26 11:00:00 | 2864.30 | 2838.11 | 2882.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 2877.60 | 2846.01 | 2881.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:30:00 | 2880.00 | 2846.01 | 2881.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 2871.30 | 2851.07 | 2880.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:30:00 | 2883.70 | 2851.07 | 2880.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 2874.00 | 2859.32 | 2875.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:30:00 | 2875.00 | 2859.32 | 2875.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 2880.00 | 2863.45 | 2875.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 11:00:00 | 2880.00 | 2863.45 | 2875.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 2878.20 | 2866.40 | 2876.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 11:30:00 | 2881.00 | 2866.40 | 2876.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 2864.40 | 2866.00 | 2875.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 13:15:00 | 2854.70 | 2866.00 | 2875.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 11:15:00 | 2905.90 | 2869.60 | 2871.09 | SL hit (close>static) qty=1.00 sl=2879.90 alert=retest2 |

### Cycle 125 — BUY (started 2025-06-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 12:15:00 | 2892.80 | 2874.24 | 2873.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 09:15:00 | 2930.10 | 2895.41 | 2884.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 14:15:00 | 2963.80 | 2965.74 | 2945.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-03 14:30:00 | 2962.30 | 2965.74 | 2945.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 2949.00 | 2974.53 | 2965.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:45:00 | 2930.00 | 2974.53 | 2965.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 2951.30 | 2969.88 | 2963.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 11:30:00 | 2967.30 | 2969.71 | 2964.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 13:00:00 | 2966.90 | 2969.15 | 2964.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 09:15:00 | 2983.00 | 2964.62 | 2963.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 2932.70 | 2958.23 | 2960.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 2932.70 | 2958.23 | 2960.66 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 12:15:00 | 2980.60 | 2963.44 | 2962.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 14:15:00 | 2994.10 | 2973.49 | 2967.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 2985.00 | 3006.09 | 2992.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 09:15:00 | 2985.00 | 3006.09 | 2992.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 2985.00 | 3006.09 | 2992.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:00:00 | 2985.00 | 3006.09 | 2992.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 2964.20 | 2997.71 | 2990.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:00:00 | 2964.20 | 2997.71 | 2990.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2025-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 12:15:00 | 2938.90 | 2979.29 | 2982.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 13:15:00 | 2930.00 | 2969.43 | 2977.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 2888.00 | 2881.66 | 2914.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 10:00:00 | 2888.00 | 2881.66 | 2914.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 2896.70 | 2873.81 | 2893.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:00:00 | 2896.70 | 2873.81 | 2893.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 2901.60 | 2879.37 | 2893.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:45:00 | 2907.10 | 2879.37 | 2893.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 2891.10 | 2881.71 | 2893.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:30:00 | 2897.30 | 2881.71 | 2893.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 2893.00 | 2883.97 | 2893.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:30:00 | 2892.70 | 2883.97 | 2893.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 2875.00 | 2882.18 | 2891.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 10:00:00 | 2856.80 | 2874.85 | 2885.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 15:15:00 | 2904.00 | 2887.15 | 2887.50 | SL hit (close>static) qty=1.00 sl=2893.00 alert=retest2 |

### Cycle 129 — BUY (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 09:15:00 | 2895.00 | 2888.72 | 2888.18 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 11:15:00 | 2884.90 | 2887.36 | 2887.62 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 12:15:00 | 2892.20 | 2888.33 | 2888.04 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 2861.80 | 2884.05 | 2886.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 10:15:00 | 2804.70 | 2868.18 | 2878.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 15:15:00 | 2740.00 | 2731.98 | 2775.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 09:15:00 | 2762.20 | 2731.98 | 2775.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 2772.60 | 2740.10 | 2775.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:45:00 | 2777.00 | 2740.10 | 2775.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 2761.40 | 2744.36 | 2773.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:15:00 | 2786.80 | 2744.36 | 2773.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 2787.80 | 2753.05 | 2775.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:45:00 | 2796.90 | 2753.05 | 2775.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 2788.00 | 2760.04 | 2776.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:30:00 | 2792.70 | 2760.04 | 2776.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 2774.80 | 2772.02 | 2777.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:45:00 | 2779.80 | 2772.02 | 2777.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 2776.80 | 2772.98 | 2777.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:45:00 | 2783.30 | 2772.98 | 2777.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 2782.00 | 2774.78 | 2777.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 2788.00 | 2774.78 | 2777.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 2765.00 | 2772.83 | 2776.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 10:30:00 | 2756.60 | 2775.42 | 2777.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 12:15:00 | 2781.40 | 2779.08 | 2778.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 12:15:00 | 2781.40 | 2779.08 | 2778.91 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 2750.20 | 2774.42 | 2777.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 2728.70 | 2765.28 | 2772.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 14:15:00 | 2580.20 | 2579.66 | 2626.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 15:00:00 | 2580.20 | 2579.66 | 2626.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 2581.60 | 2580.81 | 2598.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:30:00 | 2601.00 | 2580.81 | 2598.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 2664.00 | 2597.49 | 2602.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 13:00:00 | 2664.00 | 2597.49 | 2602.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 13:15:00 | 2691.00 | 2616.19 | 2610.13 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 2582.30 | 2605.29 | 2607.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 13:15:00 | 2573.20 | 2595.58 | 2602.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 2630.40 | 2591.78 | 2598.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 2630.40 | 2591.78 | 2598.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 2630.40 | 2591.78 | 2598.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:30:00 | 2632.70 | 2591.78 | 2598.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 2628.30 | 2599.09 | 2600.83 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2025-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 11:15:00 | 2720.00 | 2623.27 | 2611.66 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 2609.60 | 2634.72 | 2637.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 2567.90 | 2597.87 | 2614.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 2458.20 | 2452.13 | 2504.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 10:00:00 | 2458.20 | 2452.13 | 2504.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 2550.70 | 2471.85 | 2508.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:30:00 | 2580.40 | 2471.85 | 2508.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 2496.80 | 2476.84 | 2507.47 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 12:15:00 | 2529.00 | 2515.23 | 2514.86 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2025-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 12:15:00 | 2509.60 | 2515.79 | 2516.33 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 13:15:00 | 2522.00 | 2517.03 | 2516.84 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 14:15:00 | 2507.70 | 2515.17 | 2516.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 2504.30 | 2512.17 | 2514.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 10:15:00 | 2514.30 | 2512.59 | 2514.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 10:15:00 | 2514.30 | 2512.59 | 2514.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 2514.30 | 2512.59 | 2514.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:00:00 | 2514.30 | 2512.59 | 2514.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 2509.00 | 2511.87 | 2513.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 12:00:00 | 2509.00 | 2511.87 | 2513.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 2513.40 | 2512.18 | 2513.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:00:00 | 2513.40 | 2512.18 | 2513.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 2509.90 | 2511.72 | 2513.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:45:00 | 2515.40 | 2511.72 | 2513.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 2518.70 | 2513.12 | 2514.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 15:00:00 | 2518.70 | 2513.12 | 2514.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2025-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 15:15:00 | 2520.60 | 2514.62 | 2514.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 2592.60 | 2530.21 | 2521.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 2554.30 | 2570.09 | 2551.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 2554.30 | 2570.09 | 2551.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 2554.30 | 2570.09 | 2551.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:30:00 | 2552.70 | 2570.09 | 2551.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 2539.00 | 2563.87 | 2550.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:00:00 | 2539.00 | 2563.87 | 2550.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 2540.10 | 2559.12 | 2549.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:45:00 | 2546.40 | 2559.12 | 2549.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 2534.40 | 2554.17 | 2547.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:00:00 | 2534.40 | 2554.17 | 2547.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 2538.00 | 2550.94 | 2547.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:30:00 | 2535.50 | 2550.94 | 2547.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 2545.00 | 2548.59 | 2546.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 09:15:00 | 2548.90 | 2548.59 | 2546.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 09:15:00 | 2528.80 | 2544.63 | 2544.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 09:15:00 | 2528.80 | 2544.63 | 2544.94 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 14:15:00 | 2560.70 | 2547.38 | 2545.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 10:15:00 | 2599.40 | 2559.63 | 2551.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 11:15:00 | 2583.00 | 2586.64 | 2573.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 11:30:00 | 2592.90 | 2586.64 | 2573.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 2582.30 | 2589.05 | 2580.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:45:00 | 2593.00 | 2589.05 | 2580.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 2576.10 | 2586.46 | 2579.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:00:00 | 2576.10 | 2586.46 | 2579.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 2572.50 | 2583.67 | 2579.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:30:00 | 2572.30 | 2583.67 | 2579.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 2562.30 | 2579.40 | 2577.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:00:00 | 2562.30 | 2579.40 | 2577.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2025-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 14:15:00 | 2556.80 | 2572.65 | 2574.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 2542.50 | 2563.88 | 2570.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 2454.50 | 2434.66 | 2463.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 2454.50 | 2434.66 | 2463.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 2454.50 | 2434.66 | 2463.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 2459.90 | 2434.66 | 2463.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 2461.80 | 2444.46 | 2463.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:00:00 | 2461.80 | 2444.46 | 2463.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 2481.00 | 2451.77 | 2464.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:30:00 | 2480.10 | 2451.77 | 2464.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 2484.00 | 2458.21 | 2466.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:30:00 | 2481.30 | 2458.21 | 2466.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 2520.00 | 2478.20 | 2474.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 2526.90 | 2494.84 | 2483.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 13:15:00 | 2545.70 | 2546.44 | 2527.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 14:00:00 | 2545.70 | 2546.44 | 2527.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 2515.00 | 2539.57 | 2528.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:00:00 | 2515.00 | 2539.57 | 2528.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 2498.70 | 2531.39 | 2526.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 2498.70 | 2531.39 | 2526.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 2481.30 | 2521.37 | 2522.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 12:15:00 | 2470.40 | 2511.18 | 2517.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 12:15:00 | 2487.30 | 2466.33 | 2485.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 12:15:00 | 2487.30 | 2466.33 | 2485.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 2487.30 | 2466.33 | 2485.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:45:00 | 2479.50 | 2466.33 | 2485.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 2489.40 | 2470.94 | 2485.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:30:00 | 2493.00 | 2470.94 | 2485.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 2469.00 | 2470.66 | 2483.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 2504.30 | 2470.66 | 2483.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 2503.50 | 2477.23 | 2484.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:30:00 | 2509.00 | 2477.23 | 2484.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 2511.00 | 2483.98 | 2487.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:15:00 | 2513.20 | 2483.98 | 2487.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 11:15:00 | 2511.80 | 2489.55 | 2489.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 13:15:00 | 2545.50 | 2505.42 | 2497.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 12:15:00 | 2580.30 | 2582.55 | 2564.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 12:45:00 | 2579.90 | 2582.55 | 2564.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 2572.60 | 2579.88 | 2566.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 2572.60 | 2579.88 | 2566.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 2605.70 | 2585.06 | 2570.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:30:00 | 2570.00 | 2585.06 | 2570.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 2867.60 | 2843.11 | 2813.84 | EMA400 retest candle locked (from upside) |

### Cycle 150 — SELL (started 2025-09-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 15:15:00 | 2810.70 | 2824.68 | 2825.24 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 09:15:00 | 2831.90 | 2826.12 | 2825.85 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 2779.20 | 2816.74 | 2821.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 15:15:00 | 2773.00 | 2791.45 | 2805.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 2800.00 | 2761.52 | 2777.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 2800.00 | 2761.52 | 2777.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 2800.00 | 2761.52 | 2777.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:30:00 | 2786.20 | 2761.52 | 2777.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 2816.00 | 2772.42 | 2780.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 13:15:00 | 2782.80 | 2780.04 | 2782.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 13:15:00 | 2643.66 | 2697.79 | 2735.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-09-30 14:15:00 | 2504.52 | 2551.61 | 2602.36 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 153 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 2636.00 | 2609.11 | 2609.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 2736.70 | 2634.63 | 2620.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 10:15:00 | 2806.70 | 2820.57 | 2775.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 11:00:00 | 2806.70 | 2820.57 | 2775.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 2777.00 | 2805.91 | 2789.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 2777.00 | 2805.91 | 2789.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 2783.10 | 2801.35 | 2789.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 12:15:00 | 2789.00 | 2801.35 | 2789.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 12:45:00 | 2797.20 | 2801.70 | 2790.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 2792.30 | 2791.82 | 2788.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 11:15:00 | 2770.80 | 2784.35 | 2785.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — SELL (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 11:15:00 | 2770.80 | 2784.35 | 2785.44 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2025-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 14:15:00 | 2803.60 | 2787.83 | 2786.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 15:15:00 | 2850.00 | 2800.27 | 2792.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 11:15:00 | 2809.00 | 2809.93 | 2799.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-10 12:00:00 | 2809.00 | 2809.93 | 2799.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 2790.10 | 2805.96 | 2798.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 12:45:00 | 2796.30 | 2805.96 | 2798.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 2801.70 | 2805.11 | 2798.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 14:15:00 | 2806.80 | 2805.11 | 2798.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 2768.30 | 2794.94 | 2795.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 2768.30 | 2794.94 | 2795.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 2740.00 | 2783.95 | 2790.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 09:15:00 | 2752.10 | 2745.29 | 2764.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-14 09:30:00 | 2747.60 | 2745.29 | 2764.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 2753.30 | 2746.90 | 2763.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:45:00 | 2747.00 | 2746.90 | 2763.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 2745.30 | 2746.58 | 2761.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:30:00 | 2756.00 | 2746.58 | 2761.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 2752.00 | 2749.58 | 2759.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:45:00 | 2760.00 | 2749.58 | 2759.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 2760.30 | 2751.72 | 2759.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 2776.00 | 2754.16 | 2760.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 2742.00 | 2751.72 | 2758.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:30:00 | 2770.30 | 2751.72 | 2758.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 2700.00 | 2725.89 | 2743.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:30:00 | 2698.30 | 2725.89 | 2743.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 2713.80 | 2712.77 | 2729.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:30:00 | 2716.80 | 2712.77 | 2729.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 2741.50 | 2719.99 | 2729.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:00:00 | 2741.50 | 2719.99 | 2729.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 2732.20 | 2722.43 | 2729.93 | EMA400 retest candle locked (from downside) |

### Cycle 157 — BUY (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 09:15:00 | 2839.70 | 2746.94 | 2739.84 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 09:15:00 | 2775.00 | 2802.11 | 2802.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 09:15:00 | 2733.00 | 2756.79 | 2770.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 09:15:00 | 2733.10 | 2726.33 | 2745.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 09:15:00 | 2733.10 | 2726.33 | 2745.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 2733.10 | 2726.33 | 2745.20 | EMA400 retest candle locked (from downside) |

### Cycle 159 — BUY (started 2025-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 15:15:00 | 2753.20 | 2743.44 | 2743.21 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 09:15:00 | 2730.00 | 2740.76 | 2742.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 10:15:00 | 2712.00 | 2735.00 | 2739.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 14:15:00 | 2617.60 | 2605.38 | 2638.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 14:30:00 | 2626.00 | 2605.38 | 2638.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 2656.20 | 2616.62 | 2637.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 2663.80 | 2616.62 | 2637.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 2666.00 | 2626.50 | 2640.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 2656.90 | 2626.50 | 2640.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 2648.30 | 2636.75 | 2642.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 13:30:00 | 2648.00 | 2636.75 | 2642.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 2644.20 | 2638.24 | 2642.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 15:15:00 | 2639.20 | 2638.24 | 2642.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 2788.60 | 2668.46 | 2655.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — BUY (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 09:15:00 | 2788.60 | 2668.46 | 2655.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 2815.00 | 2755.61 | 2714.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 15:15:00 | 3071.00 | 3085.31 | 3030.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 09:15:00 | 3115.00 | 3085.31 | 3030.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 3077.00 | 3099.46 | 3069.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:00:00 | 3077.00 | 3099.46 | 3069.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 3078.50 | 3095.27 | 3070.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 12:15:00 | 3098.40 | 3093.66 | 3071.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 14:15:00 | 3067.00 | 3085.87 | 3073.42 | SL hit (close<static) qty=1.00 sl=3069.00 alert=retest2 |

### Cycle 162 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 3018.70 | 3091.63 | 3093.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 09:15:00 | 2965.00 | 3042.48 | 3067.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 3010.80 | 2982.59 | 3017.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 3010.80 | 2982.59 | 3017.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 3010.80 | 2982.59 | 3017.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:00:00 | 3010.80 | 2982.59 | 3017.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 3003.00 | 2986.67 | 3016.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 11:30:00 | 2981.10 | 2986.94 | 3013.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 14:30:00 | 2976.00 | 2981.71 | 3004.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 09:15:00 | 3031.50 | 2977.32 | 2976.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — BUY (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 09:15:00 | 3031.50 | 2977.32 | 2976.46 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 15:15:00 | 2960.20 | 2977.13 | 2978.27 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 2988.50 | 2979.40 | 2979.20 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 10:15:00 | 2966.80 | 2976.88 | 2978.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 11:15:00 | 2937.60 | 2969.03 | 2974.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 14:15:00 | 2901.10 | 2898.72 | 2927.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-02 15:00:00 | 2901.10 | 2898.72 | 2927.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 2898.90 | 2884.69 | 2901.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:15:00 | 2945.60 | 2884.69 | 2901.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 2945.20 | 2896.79 | 2905.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:45:00 | 2935.00 | 2896.79 | 2905.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 2928.90 | 2903.22 | 2907.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:30:00 | 2905.80 | 2909.65 | 2910.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 14:45:00 | 2906.10 | 2908.22 | 2909.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 15:15:00 | 2760.51 | 2817.04 | 2855.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 15:15:00 | 2760.79 | 2817.04 | 2855.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-08 11:15:00 | 2615.22 | 2731.06 | 2803.16 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 167 — BUY (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 12:15:00 | 2608.00 | 2597.50 | 2597.18 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 2555.40 | 2591.78 | 2594.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 2532.00 | 2568.26 | 2579.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 2506.20 | 2503.43 | 2531.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 12:00:00 | 2506.20 | 2503.43 | 2531.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 2519.30 | 2509.84 | 2523.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 2526.20 | 2509.84 | 2523.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 2525.90 | 2513.91 | 2522.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:45:00 | 2524.50 | 2513.91 | 2522.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 2543.80 | 2519.88 | 2524.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 2543.80 | 2519.88 | 2524.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 2559.20 | 2527.75 | 2527.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 2572.60 | 2540.29 | 2533.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 14:15:00 | 2676.00 | 2676.37 | 2645.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 15:00:00 | 2676.00 | 2676.37 | 2645.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 2704.30 | 2681.26 | 2652.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:45:00 | 2732.70 | 2691.43 | 2659.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 11:15:00 | 2740.70 | 2691.43 | 2659.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 14:15:00 | 2648.80 | 2667.59 | 2668.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — SELL (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 14:15:00 | 2648.80 | 2667.59 | 2668.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 2615.50 | 2655.64 | 2662.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 2614.40 | 2604.62 | 2627.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 2614.40 | 2604.62 | 2627.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 2614.40 | 2604.62 | 2627.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 2614.00 | 2604.62 | 2627.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 2626.20 | 2608.94 | 2627.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:45:00 | 2633.10 | 2608.94 | 2627.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 2615.00 | 2610.15 | 2626.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 10:00:00 | 2601.30 | 2613.64 | 2622.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 10:45:00 | 2606.00 | 2612.63 | 2620.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:30:00 | 2602.00 | 2610.59 | 2618.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 2629.60 | 2612.14 | 2615.74 | SL hit (close>static) qty=1.00 sl=2628.20 alert=retest2 |

### Cycle 171 — BUY (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 09:15:00 | 2705.50 | 2633.45 | 2624.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 11:15:00 | 2722.00 | 2660.06 | 2638.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 2684.40 | 2697.99 | 2668.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 10:00:00 | 2684.40 | 2697.99 | 2668.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 2682.00 | 2694.80 | 2669.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:30:00 | 2671.80 | 2694.80 | 2669.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 2685.00 | 2684.45 | 2673.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 2672.00 | 2684.45 | 2673.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 2670.70 | 2681.70 | 2673.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 14:30:00 | 2688.10 | 2675.65 | 2672.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 09:15:00 | 2696.90 | 2700.43 | 2692.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 13:15:00 | 2652.00 | 2682.53 | 2686.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 2652.00 | 2682.53 | 2686.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 15:15:00 | 2648.00 | 2670.66 | 2680.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 2634.00 | 2621.22 | 2643.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 2620.60 | 2621.22 | 2643.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 2611.00 | 2619.18 | 2640.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 2588.90 | 2611.28 | 2633.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:45:00 | 2585.80 | 2606.81 | 2629.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:45:00 | 2585.70 | 2601.69 | 2624.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 15:15:00 | 2585.00 | 2601.55 | 2622.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 2609.50 | 2600.49 | 2618.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:30:00 | 2615.30 | 2600.49 | 2618.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 2645.70 | 2609.53 | 2620.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 2645.70 | 2609.53 | 2620.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 2649.40 | 2617.51 | 2623.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:30:00 | 2652.40 | 2617.51 | 2623.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 2585.90 | 2612.85 | 2620.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:15:00 | 2577.00 | 2606.26 | 2616.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 2459.45 | 2490.12 | 2529.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 2456.51 | 2490.12 | 2529.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 2456.41 | 2490.12 | 2529.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 2455.75 | 2490.12 | 2529.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 2448.15 | 2490.12 | 2529.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-20 11:15:00 | 2330.01 | 2434.15 | 2496.02 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 173 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 2399.30 | 2285.22 | 2269.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 2507.70 | 2351.02 | 2303.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 2596.60 | 2671.67 | 2607.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 2596.60 | 2671.67 | 2607.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 2596.60 | 2671.67 | 2607.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 2596.60 | 2671.67 | 2607.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 2548.00 | 2646.94 | 2602.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 2533.10 | 2646.94 | 2602.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 2511.00 | 2619.75 | 2593.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 2511.00 | 2619.75 | 2593.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 2487.00 | 2566.26 | 2572.35 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 2606.70 | 2563.15 | 2561.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 11:15:00 | 2624.60 | 2588.20 | 2579.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 2537.50 | 2593.83 | 2587.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 2537.50 | 2593.83 | 2587.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 2537.50 | 2593.83 | 2587.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 2526.70 | 2593.83 | 2587.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 2546.00 | 2584.26 | 2583.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:30:00 | 2532.00 | 2584.26 | 2583.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 11:15:00 | 2538.00 | 2575.01 | 2579.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 13:15:00 | 2525.30 | 2558.48 | 2570.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 09:15:00 | 2578.90 | 2553.44 | 2564.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 2578.90 | 2553.44 | 2564.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 2578.90 | 2553.44 | 2564.65 | EMA400 retest candle locked (from downside) |

### Cycle 177 — BUY (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 10:15:00 | 2734.40 | 2589.64 | 2580.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 2767.50 | 2709.25 | 2658.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 13:15:00 | 2848.60 | 2852.93 | 2808.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 13:45:00 | 2841.90 | 2852.93 | 2808.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 2789.40 | 2833.72 | 2810.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:15:00 | 2778.40 | 2833.72 | 2810.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 2782.10 | 2823.40 | 2807.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:30:00 | 2765.00 | 2823.40 | 2807.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — SELL (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 14:15:00 | 2782.40 | 2797.16 | 2798.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 2731.20 | 2782.16 | 2791.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 12:15:00 | 2783.20 | 2770.42 | 2782.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 12:15:00 | 2783.20 | 2770.42 | 2782.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 2783.20 | 2770.42 | 2782.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 13:00:00 | 2783.20 | 2770.42 | 2782.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 13:15:00 | 2774.00 | 2771.14 | 2781.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 15:15:00 | 2755.00 | 2771.31 | 2780.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 2861.40 | 2786.72 | 2786.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — BUY (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 09:15:00 | 2861.40 | 2786.72 | 2786.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 11:15:00 | 2879.10 | 2841.20 | 2825.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 15:15:00 | 2899.00 | 2908.50 | 2882.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-20 09:15:00 | 2927.80 | 2908.50 | 2882.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 3059.30 | 2938.66 | 2898.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 11:30:00 | 3197.60 | 3129.79 | 3090.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 11:45:00 | 3192.00 | 3193.40 | 3150.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 14:30:00 | 3201.40 | 3190.83 | 3159.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 13:30:00 | 3191.50 | 3220.74 | 3191.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 3201.90 | 3213.38 | 3195.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:00:00 | 3201.90 | 3213.38 | 3195.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 3148.00 | 3200.31 | 3190.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 11:00:00 | 3148.00 | 3200.31 | 3190.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-04 11:15:00 | 3100.00 | 3180.24 | 3182.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — SELL (started 2026-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 11:15:00 | 3100.00 | 3180.24 | 3182.62 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2026-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 10:15:00 | 3309.00 | 3195.61 | 3183.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 12:15:00 | 3322.80 | 3234.78 | 3204.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 11:15:00 | 3438.50 | 3479.77 | 3406.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-09 12:00:00 | 3438.50 | 3479.77 | 3406.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 3452.40 | 3477.19 | 3445.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 12:45:00 | 3456.60 | 3477.19 | 3445.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 14:15:00 | 3441.10 | 3468.41 | 3446.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:30:00 | 3445.00 | 3468.41 | 3446.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 3439.00 | 3462.53 | 3446.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 3403.70 | 3462.53 | 3446.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 3416.00 | 3453.22 | 3443.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:30:00 | 3396.50 | 3453.22 | 3443.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 3408.40 | 3444.26 | 3440.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 10:45:00 | 3425.30 | 3444.26 | 3440.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — SELL (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 11:15:00 | 3405.20 | 3436.45 | 3437.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 12:15:00 | 3377.40 | 3424.64 | 3431.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 3355.40 | 3352.78 | 3388.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 12:00:00 | 3355.40 | 3352.78 | 3388.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 3204.00 | 3141.12 | 3201.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:45:00 | 3185.70 | 3141.12 | 3201.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 3266.10 | 3166.12 | 3207.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 3266.10 | 3166.12 | 3207.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 3248.40 | 3182.58 | 3211.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:30:00 | 3264.80 | 3182.58 | 3211.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 3303.90 | 3220.00 | 3224.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 3303.90 | 3220.00 | 3224.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 3304.00 | 3236.80 | 3231.50 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 13:15:00 | 3253.50 | 3266.93 | 3267.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 14:15:00 | 3234.90 | 3260.52 | 3264.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 3140.20 | 3109.37 | 3161.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 3140.20 | 3109.37 | 3161.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 3140.20 | 3109.37 | 3161.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 3108.50 | 3118.70 | 3160.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 3201.00 | 3135.16 | 3164.59 | SL hit (close>static) qty=1.00 sl=3183.30 alert=retest2 |

### Cycle 185 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 3236.80 | 3187.47 | 3184.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 3310.10 | 3220.42 | 3200.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 12:15:00 | 3227.60 | 3233.16 | 3212.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 12:45:00 | 3218.60 | 3233.16 | 3212.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 13:15:00 | 3232.70 | 3233.07 | 3214.00 | EMA400 retest candle locked (from upside) |

### Cycle 186 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 3145.80 | 3201.06 | 3204.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 3098.40 | 3165.20 | 3184.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 3117.30 | 3080.75 | 3121.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 3117.30 | 3080.75 | 3121.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 3117.30 | 3080.75 | 3121.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 3145.50 | 3080.75 | 3121.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 3088.90 | 3082.38 | 3118.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 2981.00 | 3102.23 | 3116.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 15:15:00 | 3053.00 | 3025.47 | 3060.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:45:00 | 3065.80 | 3042.76 | 3062.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 11:15:00 | 3142.90 | 3068.83 | 3071.28 | SL hit (close>static) qty=1.00 sl=3120.00 alert=retest2 |

### Cycle 187 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 3134.60 | 3081.98 | 3077.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 10:15:00 | 3188.10 | 3131.47 | 3105.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 10:15:00 | 3330.90 | 3359.71 | 3307.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 11:00:00 | 3330.90 | 3359.71 | 3307.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 11:15:00 | 3323.30 | 3352.43 | 3308.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 15:15:00 | 3353.00 | 3317.78 | 3306.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-21 09:15:00 | 3688.30 | 3522.55 | 3496.76 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 188 — SELL (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 10:15:00 | 4025.00 | 4055.05 | 4058.93 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 15:15:00 | 4070.00 | 4057.43 | 4057.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 4160.00 | 4077.95 | 4066.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 11:15:00 | 4194.30 | 4196.50 | 4153.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-06 12:00:00 | 4194.30 | 4196.50 | 4153.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 4178.00 | 4191.66 | 4165.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 09:15:00 | 4243.00 | 4191.66 | 4165.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 10:00:00 | 4225.80 | 4198.49 | 4170.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 12:15:00 | 4111.70 | 4172.32 | 4179.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 4111.70 | 4172.32 | 4179.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 14:15:00 | 4090.90 | 4148.18 | 4166.77 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-19 09:15:00 | 1618.65 | 2023-05-19 10:15:00 | 1596.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2023-05-25 09:15:00 | 1576.95 | 2023-05-25 14:15:00 | 1603.00 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2023-05-31 09:30:00 | 1671.70 | 2023-06-07 09:15:00 | 1838.87 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-05-31 10:00:00 | 1698.00 | 2023-06-12 13:15:00 | 1756.15 | STOP_HIT | 1.00 | 3.42% |
| BUY | retest2 | 2023-06-01 09:45:00 | 1675.60 | 2023-06-12 13:15:00 | 1756.15 | STOP_HIT | 1.00 | 4.81% |
| BUY | retest2 | 2023-06-20 12:45:00 | 1893.95 | 2023-06-22 11:15:00 | 1866.25 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2023-06-20 13:30:00 | 1895.75 | 2023-06-22 11:15:00 | 1866.25 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2023-06-20 15:15:00 | 1897.50 | 2023-06-22 11:15:00 | 1866.25 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2023-06-21 11:15:00 | 1902.65 | 2023-06-22 11:15:00 | 1866.25 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2023-06-21 12:30:00 | 1916.00 | 2023-06-22 11:15:00 | 1866.25 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2023-06-21 13:30:00 | 1918.45 | 2023-06-22 11:15:00 | 1866.25 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2023-06-21 15:15:00 | 1915.00 | 2023-06-22 11:15:00 | 1866.25 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2023-06-27 12:45:00 | 1842.45 | 2023-06-27 13:15:00 | 1844.00 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2023-07-03 09:15:00 | 1887.65 | 2023-07-10 09:15:00 | 1880.55 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2023-07-03 11:15:00 | 1882.90 | 2023-07-10 09:15:00 | 1880.55 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2023-07-04 10:30:00 | 1877.45 | 2023-07-10 09:15:00 | 1880.55 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2023-07-04 15:15:00 | 1880.25 | 2023-07-10 09:15:00 | 1880.55 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2023-07-05 11:15:00 | 1882.00 | 2023-07-10 09:15:00 | 1880.55 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2023-07-12 15:15:00 | 2045.00 | 2023-07-13 11:15:00 | 2249.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-07-14 10:15:00 | 2027.50 | 2023-07-18 12:15:00 | 2032.70 | STOP_HIT | 1.00 | 0.26% |
| BUY | retest2 | 2023-07-14 15:15:00 | 2034.80 | 2023-07-18 12:15:00 | 2032.70 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2023-07-18 12:15:00 | 2033.00 | 2023-07-18 12:15:00 | 2032.70 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2023-07-27 09:15:00 | 1987.50 | 2023-07-31 10:15:00 | 2012.15 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2023-07-27 10:00:00 | 1987.65 | 2023-07-31 10:15:00 | 2012.15 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2023-07-27 11:45:00 | 1993.90 | 2023-07-31 10:15:00 | 2012.15 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2023-07-27 12:45:00 | 1997.00 | 2023-07-31 10:15:00 | 2012.15 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2023-08-02 09:45:00 | 2022.40 | 2023-08-02 13:15:00 | 1971.90 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2023-08-02 11:15:00 | 2017.60 | 2023-08-02 13:15:00 | 1971.90 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2023-08-02 12:00:00 | 2019.00 | 2023-08-02 13:15:00 | 1971.90 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2023-08-02 13:00:00 | 2017.95 | 2023-08-02 13:15:00 | 1971.90 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2023-08-16 09:15:00 | 2122.15 | 2023-08-21 11:15:00 | 2334.37 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-08-28 15:15:00 | 2320.00 | 2023-08-31 11:15:00 | 2358.15 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2023-08-29 11:00:00 | 2323.65 | 2023-08-31 11:15:00 | 2358.15 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2023-08-30 11:45:00 | 2313.10 | 2023-08-31 11:15:00 | 2358.15 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2023-09-05 09:15:00 | 2428.80 | 2023-09-05 10:15:00 | 2397.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2023-09-13 09:30:00 | 2065.30 | 2023-09-21 15:15:00 | 2109.00 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2023-09-14 10:00:00 | 2098.00 | 2023-09-21 15:15:00 | 2109.00 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2023-09-20 09:30:00 | 2114.85 | 2023-09-21 15:15:00 | 2109.00 | STOP_HIT | 1.00 | 0.28% |
| SELL | retest2 | 2023-10-10 13:15:00 | 2068.90 | 2023-10-11 10:15:00 | 2097.95 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest1 | 2023-11-07 09:15:00 | 1969.70 | 2023-11-09 09:15:00 | 1938.30 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest1 | 2023-11-07 11:45:00 | 1965.15 | 2023-11-09 09:15:00 | 1938.30 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest1 | 2023-11-08 09:15:00 | 1967.70 | 2023-11-09 09:15:00 | 1938.30 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest1 | 2023-11-08 09:45:00 | 1967.60 | 2023-11-09 09:15:00 | 1938.30 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2023-11-15 14:30:00 | 1856.10 | 2023-11-21 09:15:00 | 1852.75 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2023-11-15 15:15:00 | 1855.00 | 2023-11-21 13:15:00 | 1852.05 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2023-11-16 10:00:00 | 1857.50 | 2023-11-21 13:15:00 | 1852.05 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest2 | 2023-11-16 10:45:00 | 1857.65 | 2023-11-21 13:15:00 | 1852.05 | STOP_HIT | 1.00 | 0.30% |
| SELL | retest2 | 2023-11-20 14:00:00 | 1834.05 | 2023-11-21 13:15:00 | 1852.05 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2023-12-14 10:30:00 | 2015.05 | 2023-12-20 14:15:00 | 1914.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-14 11:30:00 | 2012.60 | 2023-12-20 14:15:00 | 1911.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-15 11:00:00 | 2015.00 | 2023-12-20 14:15:00 | 1914.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-18 10:30:00 | 2014.85 | 2023-12-20 14:15:00 | 1914.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-14 10:30:00 | 2015.05 | 2023-12-22 09:15:00 | 1980.80 | STOP_HIT | 0.50 | 1.70% |
| SELL | retest2 | 2023-12-14 11:30:00 | 2012.60 | 2023-12-22 09:15:00 | 1980.80 | STOP_HIT | 0.50 | 1.58% |
| SELL | retest2 | 2023-12-15 11:00:00 | 2015.00 | 2023-12-22 09:15:00 | 1980.80 | STOP_HIT | 0.50 | 1.70% |
| SELL | retest2 | 2023-12-18 10:30:00 | 2014.85 | 2023-12-22 09:15:00 | 1980.80 | STOP_HIT | 0.50 | 1.69% |
| SELL | retest2 | 2023-12-19 14:30:00 | 1990.10 | 2023-12-27 11:15:00 | 1890.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-20 11:45:00 | 1992.10 | 2023-12-27 11:15:00 | 1892.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-20 12:15:00 | 1992.60 | 2023-12-27 11:15:00 | 1892.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-19 14:30:00 | 1990.10 | 2024-01-02 09:15:00 | 1912.20 | STOP_HIT | 0.50 | 3.91% |
| SELL | retest2 | 2023-12-20 11:45:00 | 1992.10 | 2024-01-02 09:15:00 | 1912.20 | STOP_HIT | 0.50 | 4.01% |
| SELL | retest2 | 2023-12-20 12:15:00 | 1992.60 | 2024-01-02 09:15:00 | 1912.20 | STOP_HIT | 0.50 | 4.03% |
| BUY | retest2 | 2024-01-08 10:15:00 | 1917.50 | 2024-01-15 14:15:00 | 1958.00 | STOP_HIT | 1.00 | 2.11% |
| BUY | retest2 | 2024-02-01 09:15:00 | 1989.90 | 2024-02-02 14:15:00 | 1912.10 | STOP_HIT | 1.00 | -3.91% |
| SELL | retest2 | 2024-02-07 11:00:00 | 1897.25 | 2024-02-08 11:15:00 | 1903.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2024-02-07 11:45:00 | 1892.10 | 2024-02-08 11:15:00 | 1903.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-02-07 12:45:00 | 1893.50 | 2024-02-08 11:15:00 | 1903.00 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2024-02-07 13:15:00 | 1894.10 | 2024-02-08 11:15:00 | 1903.00 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2024-02-16 09:15:00 | 1963.95 | 2024-02-19 09:15:00 | 2160.35 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-03-11 09:15:00 | 2640.00 | 2024-03-12 09:15:00 | 2508.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-11 09:15:00 | 2640.00 | 2024-03-13 09:15:00 | 2376.00 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-04-01 09:15:00 | 2503.65 | 2024-04-02 09:15:00 | 2754.02 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-04-30 15:15:00 | 2962.00 | 2024-05-06 09:15:00 | 2813.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-02 09:45:00 | 2963.40 | 2024-05-06 09:15:00 | 2815.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-02 10:45:00 | 2970.00 | 2024-05-06 09:15:00 | 2821.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-30 15:15:00 | 2962.00 | 2024-05-08 09:15:00 | 2828.00 | STOP_HIT | 0.50 | 4.52% |
| SELL | retest2 | 2024-05-02 09:45:00 | 2963.40 | 2024-05-08 09:15:00 | 2828.00 | STOP_HIT | 0.50 | 4.57% |
| SELL | retest2 | 2024-05-02 10:45:00 | 2970.00 | 2024-05-08 09:15:00 | 2828.00 | STOP_HIT | 0.50 | 4.78% |
| BUY | retest2 | 2024-05-16 09:15:00 | 2893.65 | 2024-05-17 13:15:00 | 3183.02 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-05-24 13:15:00 | 3064.00 | 2024-05-27 12:15:00 | 2910.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-24 13:15:00 | 3064.00 | 2024-05-29 09:15:00 | 2919.90 | STOP_HIT | 0.50 | 4.70% |
| BUY | retest2 | 2024-06-03 09:15:00 | 3005.00 | 2024-06-04 09:15:00 | 2824.95 | STOP_HIT | 1.00 | -5.99% |
| SELL | retest2 | 2024-06-06 12:00:00 | 2627.05 | 2024-06-10 13:15:00 | 2670.00 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-06-06 13:45:00 | 2631.90 | 2024-06-10 13:15:00 | 2670.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2024-06-07 10:00:00 | 2628.55 | 2024-06-10 13:15:00 | 2670.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2024-06-07 10:45:00 | 2633.35 | 2024-06-10 13:15:00 | 2670.00 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-06-12 09:15:00 | 2724.55 | 2024-06-18 09:15:00 | 2997.01 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-24 11:30:00 | 2940.00 | 2024-06-24 13:15:00 | 2992.00 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-06-26 09:15:00 | 3009.55 | 2024-07-05 09:15:00 | 3290.05 | TARGET_HIT | 1.00 | 9.32% |
| BUY | retest2 | 2024-06-26 13:30:00 | 2990.95 | 2024-07-05 09:15:00 | 3290.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-26 15:15:00 | 2991.00 | 2024-07-05 10:15:00 | 3310.51 | TARGET_HIT | 1.00 | 10.68% |
| SELL | retest2 | 2024-07-22 14:45:00 | 3190.00 | 2024-07-23 12:15:00 | 3030.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-23 09:15:00 | 3182.35 | 2024-07-23 12:15:00 | 3023.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-23 11:15:00 | 3175.40 | 2024-07-23 12:15:00 | 3016.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-22 14:45:00 | 3190.00 | 2024-07-24 15:15:00 | 3111.00 | STOP_HIT | 0.50 | 2.48% |
| SELL | retest2 | 2024-07-23 09:15:00 | 3182.35 | 2024-07-24 15:15:00 | 3111.00 | STOP_HIT | 0.50 | 2.24% |
| SELL | retest2 | 2024-07-23 11:15:00 | 3175.40 | 2024-07-24 15:15:00 | 3111.00 | STOP_HIT | 0.50 | 2.03% |
| SELL | retest2 | 2024-08-06 10:30:00 | 2978.10 | 2024-08-13 11:15:00 | 3008.90 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-08-06 11:00:00 | 2971.70 | 2024-08-13 12:15:00 | 3042.00 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2024-08-08 10:00:00 | 2989.55 | 2024-08-13 12:15:00 | 3042.00 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-08-08 13:15:00 | 2978.65 | 2024-08-13 12:15:00 | 3042.00 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-08-09 10:15:00 | 2953.10 | 2024-08-13 12:15:00 | 3042.00 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2024-08-09 13:30:00 | 2962.85 | 2024-08-13 12:15:00 | 3042.00 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2024-08-09 14:30:00 | 2958.10 | 2024-08-13 12:15:00 | 3042.00 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2024-08-09 15:15:00 | 2956.00 | 2024-08-13 12:15:00 | 3042.00 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2024-08-12 09:15:00 | 2916.45 | 2024-08-13 12:15:00 | 3042.00 | STOP_HIT | 1.00 | -4.30% |
| SELL | retest2 | 2024-08-21 14:15:00 | 2870.35 | 2024-08-23 14:15:00 | 2885.05 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-08-22 13:00:00 | 2870.75 | 2024-08-23 14:15:00 | 2885.05 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2024-08-22 14:30:00 | 2870.00 | 2024-08-23 14:15:00 | 2885.05 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2024-08-22 15:15:00 | 2869.00 | 2024-08-23 14:15:00 | 2885.05 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2024-08-23 11:30:00 | 2879.00 | 2024-08-23 14:15:00 | 2885.05 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2024-08-23 14:15:00 | 2877.40 | 2024-08-23 14:15:00 | 2885.05 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2024-08-28 12:45:00 | 2862.45 | 2024-09-05 10:15:00 | 2793.10 | STOP_HIT | 1.00 | 2.42% |
| SELL | retest2 | 2024-08-29 09:15:00 | 2852.10 | 2024-09-05 10:15:00 | 2793.10 | STOP_HIT | 1.00 | 2.07% |
| SELL | retest2 | 2024-09-11 09:15:00 | 2708.15 | 2024-09-16 14:15:00 | 2749.95 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-09-23 11:45:00 | 2549.55 | 2024-09-27 09:15:00 | 2422.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-23 13:30:00 | 2548.00 | 2024-09-27 09:15:00 | 2420.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-24 09:15:00 | 2508.40 | 2024-09-27 14:15:00 | 2382.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-23 11:45:00 | 2549.55 | 2024-10-01 10:15:00 | 2344.60 | STOP_HIT | 0.50 | 8.04% |
| SELL | retest2 | 2024-09-23 13:30:00 | 2548.00 | 2024-10-01 10:15:00 | 2344.60 | STOP_HIT | 0.50 | 7.98% |
| SELL | retest2 | 2024-09-24 09:15:00 | 2508.40 | 2024-10-01 10:15:00 | 2344.60 | STOP_HIT | 0.50 | 6.53% |
| SELL | retest2 | 2024-10-18 15:00:00 | 2451.70 | 2024-10-22 09:15:00 | 2329.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 09:15:00 | 2425.30 | 2024-10-22 10:15:00 | 2304.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 15:00:00 | 2451.70 | 2024-10-23 09:15:00 | 2206.53 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-21 09:15:00 | 2425.30 | 2024-10-23 09:15:00 | 2182.77 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-11-04 11:45:00 | 2382.05 | 2024-11-05 10:15:00 | 2341.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-11-04 12:30:00 | 2373.95 | 2024-11-05 10:15:00 | 2341.00 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-11-04 13:00:00 | 2376.30 | 2024-11-05 10:15:00 | 2341.00 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-11-12 10:30:00 | 2222.95 | 2024-11-19 10:15:00 | 2300.00 | STOP_HIT | 1.00 | -3.47% |
| SELL | retest2 | 2024-11-13 09:15:00 | 2206.20 | 2024-11-19 10:15:00 | 2300.00 | STOP_HIT | 1.00 | -4.25% |
| BUY | retest2 | 2024-12-04 09:15:00 | 2598.70 | 2024-12-12 12:15:00 | 2611.75 | STOP_HIT | 1.00 | 0.50% |
| SELL | retest2 | 2024-12-20 12:30:00 | 2529.80 | 2024-12-27 13:15:00 | 2535.15 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2024-12-20 13:30:00 | 2531.90 | 2024-12-27 13:15:00 | 2535.15 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2024-12-20 14:45:00 | 2522.25 | 2024-12-27 13:15:00 | 2535.15 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-12-23 09:15:00 | 2468.50 | 2024-12-27 13:15:00 | 2535.15 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2024-12-24 14:15:00 | 2492.00 | 2024-12-27 13:15:00 | 2535.15 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2024-12-24 14:45:00 | 2495.00 | 2024-12-27 13:15:00 | 2535.15 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2024-12-26 11:30:00 | 2493.65 | 2024-12-27 13:15:00 | 2535.15 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-12-27 09:30:00 | 2492.85 | 2024-12-27 13:15:00 | 2535.15 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-01-02 14:45:00 | 2494.50 | 2025-01-06 09:15:00 | 2444.75 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-01-03 09:15:00 | 2520.00 | 2025-01-06 09:15:00 | 2444.75 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2025-01-08 11:00:00 | 2352.50 | 2025-01-10 09:15:00 | 2234.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 11:30:00 | 2352.40 | 2025-01-10 09:15:00 | 2234.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 11:00:00 | 2352.50 | 2025-01-10 10:15:00 | 2316.95 | STOP_HIT | 0.50 | 1.51% |
| SELL | retest2 | 2025-01-08 11:30:00 | 2352.40 | 2025-01-10 10:15:00 | 2316.95 | STOP_HIT | 0.50 | 1.51% |
| SELL | retest2 | 2025-01-23 14:00:00 | 2189.20 | 2025-01-27 09:15:00 | 2079.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 14:00:00 | 2189.20 | 2025-01-28 09:15:00 | 1970.28 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-01-31 10:15:00 | 2185.00 | 2025-02-01 12:15:00 | 2072.55 | STOP_HIT | 1.00 | -5.15% |
| SELL | retest2 | 2025-02-06 09:15:00 | 1973.75 | 2025-02-10 10:15:00 | 1875.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 09:15:00 | 1973.75 | 2025-02-11 11:15:00 | 1776.38 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-04 11:30:00 | 1404.20 | 2025-03-07 09:15:00 | 1521.30 | STOP_HIT | 1.00 | -8.34% |
| SELL | retest2 | 2025-03-05 10:00:00 | 1400.75 | 2025-03-07 09:15:00 | 1521.30 | STOP_HIT | 1.00 | -8.61% |
| SELL | retest2 | 2025-03-05 14:30:00 | 1406.80 | 2025-03-07 09:15:00 | 1521.30 | STOP_HIT | 1.00 | -8.14% |
| BUY | retest2 | 2025-03-12 09:15:00 | 1605.20 | 2025-03-19 11:15:00 | 1753.40 | TARGET_HIT | 1.00 | 9.23% |
| BUY | retest2 | 2025-03-12 13:00:00 | 1594.00 | 2025-03-19 12:15:00 | 1765.72 | TARGET_HIT | 1.00 | 10.77% |
| BUY | retest2 | 2025-03-13 09:45:00 | 1666.45 | 2025-03-26 11:15:00 | 1715.00 | STOP_HIT | 1.00 | 2.91% |
| SELL | retest2 | 2025-03-28 11:15:00 | 1707.35 | 2025-04-02 13:15:00 | 1705.00 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2025-04-01 14:15:00 | 1700.50 | 2025-04-02 13:15:00 | 1705.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-04-07 09:15:00 | 1597.50 | 2025-04-07 09:15:00 | 1437.75 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-08 10:30:00 | 1644.45 | 2025-04-11 12:15:00 | 1665.55 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-04-09 09:15:00 | 1623.60 | 2025-04-11 12:15:00 | 1665.55 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2025-04-11 09:30:00 | 1651.45 | 2025-04-11 12:15:00 | 1665.55 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-04-11 10:15:00 | 1649.00 | 2025-04-11 12:15:00 | 1665.55 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-04-28 10:30:00 | 2268.00 | 2025-04-29 12:15:00 | 2494.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-28 14:45:00 | 2247.00 | 2025-04-29 12:15:00 | 2471.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-29 09:15:00 | 2307.70 | 2025-04-29 14:15:00 | 2538.47 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-08 13:30:00 | 2219.20 | 2025-05-09 11:15:00 | 2284.20 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2025-06-02 09:45:00 | 2930.20 | 2025-06-06 09:15:00 | 3223.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-02 10:45:00 | 2923.00 | 2025-06-06 09:15:00 | 3215.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-16 09:30:00 | 2930.50 | 2025-06-17 09:15:00 | 3023.90 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2025-06-16 12:30:00 | 2971.80 | 2025-06-17 09:15:00 | 3023.90 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-06-27 13:15:00 | 2854.70 | 2025-06-30 11:15:00 | 2905.90 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-07-07 11:30:00 | 2967.30 | 2025-07-08 09:15:00 | 2932.70 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-07-07 13:00:00 | 2966.90 | 2025-07-08 09:15:00 | 2932.70 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-07-08 09:15:00 | 2983.00 | 2025-07-08 09:15:00 | 2932.70 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-07-16 10:00:00 | 2856.80 | 2025-07-16 15:15:00 | 2904.00 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-07-24 10:30:00 | 2756.60 | 2025-07-24 12:15:00 | 2781.40 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-08-20 09:15:00 | 2548.90 | 2025-08-20 09:15:00 | 2528.80 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-09-25 13:15:00 | 2782.80 | 2025-09-26 13:15:00 | 2643.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 13:15:00 | 2782.80 | 2025-09-30 14:15:00 | 2504.52 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-10-08 12:15:00 | 2789.00 | 2025-10-09 11:15:00 | 2770.80 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-10-08 12:45:00 | 2797.20 | 2025-10-09 11:15:00 | 2770.80 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-10-09 09:15:00 | 2792.30 | 2025-10-09 11:15:00 | 2770.80 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-10-10 14:15:00 | 2806.80 | 2025-10-13 09:15:00 | 2768.30 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-11-10 15:15:00 | 2639.20 | 2025-11-11 09:15:00 | 2788.60 | STOP_HIT | 1.00 | -5.66% |
| BUY | retest2 | 2025-11-19 12:15:00 | 3098.40 | 2025-11-19 14:15:00 | 3067.00 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-11-20 09:15:00 | 3134.90 | 2025-11-21 09:15:00 | 3062.40 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-11-25 11:30:00 | 2981.10 | 2025-11-28 09:15:00 | 3031.50 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-11-25 14:30:00 | 2976.00 | 2025-11-28 09:15:00 | 3031.50 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-12-04 13:30:00 | 2905.80 | 2025-12-05 15:15:00 | 2760.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 14:45:00 | 2906.10 | 2025-12-05 15:15:00 | 2760.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 13:30:00 | 2905.80 | 2025-12-08 11:15:00 | 2615.22 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-04 14:45:00 | 2906.10 | 2025-12-08 11:15:00 | 2615.49 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-12-26 10:45:00 | 2732.70 | 2025-12-29 14:15:00 | 2648.80 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2025-12-26 11:15:00 | 2740.70 | 2025-12-29 14:15:00 | 2648.80 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2026-01-01 10:00:00 | 2601.30 | 2026-01-02 10:15:00 | 2629.60 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-01-01 10:45:00 | 2606.00 | 2026-01-02 10:15:00 | 2629.60 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-01-01 12:30:00 | 2602.00 | 2026-01-02 10:15:00 | 2629.60 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-01-02 14:15:00 | 2604.00 | 2026-01-05 09:15:00 | 2705.50 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest2 | 2026-01-07 14:30:00 | 2688.10 | 2026-01-09 13:15:00 | 2652.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2026-01-09 09:15:00 | 2696.90 | 2026-01-09 13:15:00 | 2652.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2026-01-13 12:00:00 | 2588.90 | 2026-01-20 09:15:00 | 2459.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:45:00 | 2585.80 | 2026-01-20 09:15:00 | 2456.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 13:45:00 | 2585.70 | 2026-01-20 09:15:00 | 2456.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 15:15:00 | 2585.00 | 2026-01-20 09:15:00 | 2455.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 15:15:00 | 2577.00 | 2026-01-20 09:15:00 | 2448.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:00:00 | 2588.90 | 2026-01-20 11:15:00 | 2330.01 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-13 12:45:00 | 2585.80 | 2026-01-20 11:15:00 | 2327.22 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-13 13:45:00 | 2585.70 | 2026-01-20 11:15:00 | 2327.13 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-13 15:15:00 | 2585.00 | 2026-01-20 11:15:00 | 2326.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-14 15:15:00 | 2577.00 | 2026-01-20 11:15:00 | 2319.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-13 15:15:00 | 2755.00 | 2026-02-16 09:15:00 | 2861.40 | STOP_HIT | 1.00 | -3.86% |
| BUY | retest2 | 2026-02-26 11:30:00 | 3197.60 | 2026-03-04 11:15:00 | 3100.00 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2026-02-27 11:45:00 | 3192.00 | 2026-03-04 11:15:00 | 3100.00 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2026-02-27 14:30:00 | 3201.40 | 2026-03-04 11:15:00 | 3100.00 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2026-03-02 13:30:00 | 3191.50 | 2026-03-04 11:15:00 | 3100.00 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2026-03-24 10:30:00 | 3108.50 | 2026-03-24 11:15:00 | 3201.00 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2026-04-02 09:15:00 | 2981.00 | 2026-04-06 11:15:00 | 3142.90 | STOP_HIT | 1.00 | -5.43% |
| SELL | retest2 | 2026-04-02 15:15:00 | 3053.00 | 2026-04-06 11:15:00 | 3142.90 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2026-04-06 09:45:00 | 3065.80 | 2026-04-06 11:15:00 | 3142.90 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2026-04-13 15:15:00 | 3353.00 | 2026-04-21 09:15:00 | 3688.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-05-07 09:15:00 | 4243.00 | 2026-05-08 12:15:00 | 4111.70 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2026-05-07 10:00:00 | 4225.80 | 2026-05-08 12:15:00 | 4111.70 | STOP_HIT | 1.00 | -2.70% |
