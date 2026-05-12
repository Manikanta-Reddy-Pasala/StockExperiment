# PB Fintech Ltd. (POLICYBZR)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1647.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 86 |
| ALERT1 | 58 |
| ALERT2 | 57 |
| ALERT2_SKIP | 26 |
| ALERT3 | 153 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 59 |
| PARTIAL | 6 |
| TARGET_HIT | 0 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 65 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 48
- **Target hits / Stop hits / Partials:** 0 / 59 / 6
- **Avg / median % per leg:** -0.22% / -1.27%
- **Sum % (uncompounded):** -14.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 29 | 4 | 13.8% | 0 | 29 | 0 | -1.00% | -28.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 29 | 4 | 13.8% | 0 | 29 | 0 | -1.00% | -28.9% |
| SELL (all) | 36 | 13 | 36.1% | 0 | 30 | 6 | 0.40% | 14.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 36 | 13 | 36.1% | 0 | 30 | 6 | 0.40% | 14.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 65 | 17 | 26.2% | 0 | 59 | 6 | -0.22% | -14.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 1667.20 | 1629.82 | 1628.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 1709.10 | 1661.60 | 1645.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 09:15:00 | 1663.70 | 1671.06 | 1654.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 10:00:00 | 1663.70 | 1671.06 | 1654.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 1759.70 | 1773.26 | 1757.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 14:45:00 | 1741.40 | 1773.26 | 1757.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 15:15:00 | 1755.40 | 1769.68 | 1757.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:15:00 | 1716.20 | 1769.68 | 1757.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 1685.70 | 1752.89 | 1750.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:00:00 | 1685.70 | 1752.89 | 1750.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 10:15:00 | 1696.10 | 1741.53 | 1745.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 09:15:00 | 1662.30 | 1701.65 | 1721.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 1698.30 | 1670.09 | 1690.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 1698.30 | 1670.09 | 1690.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1698.30 | 1670.09 | 1690.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:00:00 | 1698.30 | 1670.09 | 1690.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1723.50 | 1680.77 | 1693.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 1723.50 | 1680.77 | 1693.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 14:15:00 | 1726.90 | 1702.89 | 1701.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 09:15:00 | 1747.80 | 1715.34 | 1707.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 15:15:00 | 1786.50 | 1788.65 | 1772.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 09:15:00 | 1769.00 | 1788.65 | 1772.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1756.60 | 1782.24 | 1771.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 1756.60 | 1782.24 | 1771.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 1779.70 | 1781.73 | 1772.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 1793.90 | 1781.73 | 1772.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 14:15:00 | 1780.90 | 1781.89 | 1774.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 10:45:00 | 1781.60 | 1777.56 | 1774.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 12:15:00 | 1783.20 | 1777.25 | 1774.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 1777.10 | 1777.22 | 1775.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 13:15:00 | 1771.10 | 1777.22 | 1775.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 1762.00 | 1774.18 | 1773.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:00:00 | 1762.00 | 1774.18 | 1773.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-28 14:15:00 | 1746.30 | 1768.60 | 1771.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 14:15:00 | 1746.30 | 1768.60 | 1771.39 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 12:15:00 | 1786.40 | 1773.32 | 1772.28 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 10:15:00 | 1751.70 | 1769.54 | 1771.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 09:15:00 | 1730.00 | 1754.00 | 1760.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 1782.30 | 1750.44 | 1753.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 1782.30 | 1750.44 | 1753.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 1782.30 | 1750.44 | 1753.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:00:00 | 1782.30 | 1750.44 | 1753.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 10:15:00 | 1785.60 | 1757.47 | 1756.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 11:15:00 | 1799.20 | 1765.81 | 1760.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 13:15:00 | 1911.30 | 1922.40 | 1893.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 14:00:00 | 1911.30 | 1922.40 | 1893.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 1901.00 | 1916.55 | 1895.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:15:00 | 1884.00 | 1916.55 | 1895.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 1891.00 | 1911.44 | 1895.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:30:00 | 1878.80 | 1911.44 | 1895.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 1895.70 | 1908.29 | 1895.20 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-06-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 14:15:00 | 1873.30 | 1889.56 | 1889.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 09:15:00 | 1860.30 | 1881.06 | 1885.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 14:15:00 | 1890.60 | 1878.57 | 1881.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 14:15:00 | 1890.60 | 1878.57 | 1881.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 1890.60 | 1878.57 | 1881.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:00:00 | 1890.60 | 1878.57 | 1881.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 1895.00 | 1881.85 | 1883.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 1902.50 | 1881.85 | 1883.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 09:15:00 | 1912.10 | 1887.90 | 1885.78 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 1870.70 | 1883.61 | 1884.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 1860.00 | 1875.43 | 1880.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 1873.50 | 1871.10 | 1876.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 13:15:00 | 1873.50 | 1871.10 | 1876.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 13:15:00 | 1873.50 | 1871.10 | 1876.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 13:45:00 | 1874.00 | 1871.10 | 1876.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 1893.20 | 1875.52 | 1877.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 1893.20 | 1875.52 | 1877.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 1892.00 | 1878.82 | 1879.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 1869.90 | 1878.82 | 1879.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 13:15:00 | 1909.10 | 1881.42 | 1879.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 13:15:00 | 1909.10 | 1881.42 | 1879.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 1949.80 | 1902.98 | 1890.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 09:15:00 | 1918.70 | 1934.85 | 1917.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 1918.70 | 1934.85 | 1917.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 1918.70 | 1934.85 | 1917.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:15:00 | 1906.20 | 1934.85 | 1917.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 1902.50 | 1928.38 | 1915.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:00:00 | 1902.50 | 1928.38 | 1915.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 1888.10 | 1920.32 | 1913.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:30:00 | 1887.10 | 1920.32 | 1913.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 13:15:00 | 1886.80 | 1907.18 | 1908.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 1873.30 | 1891.76 | 1899.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 1879.30 | 1879.01 | 1889.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 09:45:00 | 1877.60 | 1879.01 | 1889.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1887.50 | 1880.71 | 1889.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:45:00 | 1889.70 | 1880.71 | 1889.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1889.70 | 1882.50 | 1889.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:30:00 | 1892.60 | 1882.50 | 1889.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 1891.20 | 1884.24 | 1889.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:30:00 | 1893.20 | 1884.24 | 1889.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 1899.70 | 1887.33 | 1890.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:45:00 | 1902.00 | 1887.33 | 1890.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 1895.90 | 1891.27 | 1892.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 1886.20 | 1891.27 | 1892.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 11:15:00 | 1903.10 | 1892.63 | 1892.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 1903.10 | 1892.63 | 1892.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 1915.10 | 1900.17 | 1896.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 10:15:00 | 1888.20 | 1897.77 | 1895.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 10:15:00 | 1888.20 | 1897.77 | 1895.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 1888.20 | 1897.77 | 1895.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:00:00 | 1888.20 | 1897.77 | 1895.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 1880.30 | 1894.28 | 1894.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 12:00:00 | 1880.30 | 1894.28 | 1894.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 12:15:00 | 1888.10 | 1893.04 | 1893.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-24 13:15:00 | 1869.50 | 1888.33 | 1891.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 14:15:00 | 1833.60 | 1832.29 | 1848.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-26 15:00:00 | 1833.60 | 1832.29 | 1848.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1863.90 | 1838.25 | 1848.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:00:00 | 1863.90 | 1838.25 | 1848.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 1852.80 | 1841.16 | 1848.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 12:00:00 | 1841.50 | 1841.23 | 1848.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 11:15:00 | 1845.40 | 1829.68 | 1828.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 11:15:00 | 1845.40 | 1829.68 | 1828.46 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 13:15:00 | 1804.40 | 1823.08 | 1825.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 1773.40 | 1802.79 | 1812.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 15:15:00 | 1798.00 | 1796.02 | 1805.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-07 09:15:00 | 1815.80 | 1796.02 | 1805.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1805.00 | 1797.81 | 1805.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 1801.70 | 1797.81 | 1805.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 1804.30 | 1799.11 | 1805.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:15:00 | 1794.10 | 1799.11 | 1805.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 14:15:00 | 1802.20 | 1795.58 | 1801.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 14:15:00 | 1827.80 | 1802.02 | 1804.27 | SL hit (close>static) qty=1.00 sl=1810.20 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 15:15:00 | 1827.90 | 1807.20 | 1806.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 12:15:00 | 1835.00 | 1817.24 | 1811.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 13:15:00 | 1843.20 | 1845.46 | 1832.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 14:00:00 | 1843.20 | 1845.46 | 1832.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1859.10 | 1851.42 | 1839.08 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 09:15:00 | 1798.00 | 1827.56 | 1831.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 1789.20 | 1819.89 | 1827.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 11:15:00 | 1811.60 | 1799.96 | 1809.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 11:15:00 | 1811.60 | 1799.96 | 1809.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 1811.60 | 1799.96 | 1809.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:30:00 | 1808.80 | 1799.96 | 1809.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 1812.60 | 1802.49 | 1810.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:00:00 | 1812.60 | 1802.49 | 1810.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 1823.90 | 1806.77 | 1811.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:00:00 | 1823.90 | 1806.77 | 1811.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 1833.10 | 1812.04 | 1813.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:45:00 | 1833.70 | 1812.04 | 1813.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 15:15:00 | 1834.00 | 1816.43 | 1815.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 1852.00 | 1823.54 | 1818.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 1843.60 | 1847.11 | 1835.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 1843.60 | 1847.11 | 1835.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 1843.60 | 1847.11 | 1835.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:45:00 | 1840.50 | 1847.11 | 1835.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 1838.20 | 1844.17 | 1836.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:30:00 | 1833.00 | 1844.17 | 1836.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 1828.50 | 1841.04 | 1835.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:45:00 | 1830.60 | 1841.04 | 1835.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 1814.10 | 1835.65 | 1833.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:00:00 | 1814.10 | 1835.65 | 1833.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 14:15:00 | 1810.00 | 1830.52 | 1831.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 09:15:00 | 1797.40 | 1821.06 | 1826.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 10:15:00 | 1767.40 | 1766.77 | 1782.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 11:00:00 | 1767.40 | 1766.77 | 1782.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1780.00 | 1769.42 | 1782.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:00:00 | 1780.00 | 1769.42 | 1782.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 1799.20 | 1775.37 | 1783.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:00:00 | 1799.20 | 1775.37 | 1783.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 1786.40 | 1777.58 | 1783.94 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 09:15:00 | 1797.50 | 1788.96 | 1788.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 11:15:00 | 1823.80 | 1798.85 | 1793.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 14:15:00 | 1816.50 | 1817.29 | 1810.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-23 14:30:00 | 1817.30 | 1817.29 | 1810.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 1766.50 | 1808.25 | 1807.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:00:00 | 1766.50 | 1808.25 | 1807.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 1791.70 | 1804.94 | 1805.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 12:15:00 | 1745.10 | 1773.87 | 1783.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 14:15:00 | 1782.70 | 1773.69 | 1781.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 14:15:00 | 1782.70 | 1773.69 | 1781.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 14:15:00 | 1782.70 | 1773.69 | 1781.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 15:00:00 | 1782.70 | 1773.69 | 1781.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 1790.00 | 1776.95 | 1782.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:15:00 | 1769.90 | 1776.95 | 1782.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 1772.10 | 1775.98 | 1781.48 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 15:15:00 | 1794.00 | 1785.14 | 1784.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 1814.60 | 1791.03 | 1786.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 1789.50 | 1809.80 | 1801.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 1789.50 | 1809.80 | 1801.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1789.50 | 1809.80 | 1801.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:15:00 | 1782.40 | 1809.80 | 1801.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 1792.10 | 1806.26 | 1800.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:45:00 | 1802.40 | 1804.83 | 1800.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 1779.50 | 1802.10 | 1801.16 | SL hit (close<static) qty=1.00 sl=1780.50 alert=retest2 |

### Cycle 24 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 1780.80 | 1796.97 | 1798.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 12:15:00 | 1778.40 | 1793.26 | 1797.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 13:15:00 | 1776.70 | 1775.35 | 1783.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 13:45:00 | 1780.30 | 1775.35 | 1783.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1782.30 | 1776.74 | 1783.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 1782.30 | 1776.74 | 1783.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 1778.20 | 1777.03 | 1783.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 09:15:00 | 1764.10 | 1777.03 | 1783.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 15:15:00 | 1786.30 | 1760.20 | 1757.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 1786.30 | 1760.20 | 1757.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 10:15:00 | 1813.00 | 1779.53 | 1769.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 11:15:00 | 1824.60 | 1829.50 | 1807.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 12:00:00 | 1824.60 | 1829.50 | 1807.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 1841.90 | 1850.22 | 1837.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:30:00 | 1847.10 | 1850.22 | 1837.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 1840.60 | 1848.29 | 1837.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:30:00 | 1841.20 | 1848.29 | 1837.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 1840.30 | 1846.69 | 1837.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 1899.90 | 1839.56 | 1836.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 10:15:00 | 1897.80 | 1903.58 | 1904.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 10:15:00 | 1897.80 | 1903.58 | 1904.15 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 11:15:00 | 1910.30 | 1904.92 | 1904.71 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 15:15:00 | 1898.30 | 1903.83 | 1904.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 1864.80 | 1896.03 | 1900.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 1803.70 | 1792.55 | 1813.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 09:45:00 | 1804.10 | 1792.55 | 1813.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 1818.00 | 1802.56 | 1810.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 1818.00 | 1802.56 | 1810.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 1812.00 | 1804.45 | 1810.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 1805.90 | 1804.45 | 1810.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1818.90 | 1807.34 | 1811.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:45:00 | 1819.80 | 1807.34 | 1811.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 1831.80 | 1812.23 | 1813.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 11:00:00 | 1831.80 | 1812.23 | 1813.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 1830.60 | 1815.91 | 1815.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 1834.50 | 1823.43 | 1819.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 14:15:00 | 1831.20 | 1832.26 | 1825.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 14:15:00 | 1831.20 | 1832.26 | 1825.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 1831.20 | 1832.26 | 1825.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 15:00:00 | 1831.20 | 1832.26 | 1825.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1853.00 | 1863.13 | 1852.06 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2025-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 13:15:00 | 1833.20 | 1846.77 | 1846.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 14:15:00 | 1821.40 | 1841.70 | 1844.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 13:15:00 | 1801.10 | 1797.48 | 1810.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 14:00:00 | 1801.10 | 1797.48 | 1810.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 1804.00 | 1798.78 | 1809.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:45:00 | 1809.10 | 1798.78 | 1809.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 1810.10 | 1801.05 | 1809.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 1817.50 | 1801.05 | 1809.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1818.60 | 1804.56 | 1810.44 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 12:15:00 | 1825.10 | 1815.13 | 1814.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 15:15:00 | 1837.90 | 1823.17 | 1818.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 1810.20 | 1820.58 | 1817.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 1810.20 | 1820.58 | 1817.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 1810.20 | 1820.58 | 1817.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:00:00 | 1810.20 | 1820.58 | 1817.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 1814.90 | 1819.44 | 1817.54 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 12:15:00 | 1806.00 | 1814.94 | 1815.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 12:15:00 | 1800.50 | 1809.67 | 1812.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 12:15:00 | 1804.40 | 1798.15 | 1804.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 12:15:00 | 1804.40 | 1798.15 | 1804.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 1804.40 | 1798.15 | 1804.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:00:00 | 1804.40 | 1798.15 | 1804.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 1812.80 | 1801.08 | 1805.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:00:00 | 1812.80 | 1801.08 | 1805.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 1806.20 | 1802.11 | 1805.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:30:00 | 1814.10 | 1802.11 | 1805.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 1826.80 | 1807.35 | 1807.07 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 1794.80 | 1810.05 | 1811.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 14:15:00 | 1793.30 | 1806.70 | 1810.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 09:15:00 | 1791.10 | 1789.19 | 1796.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 1791.10 | 1789.19 | 1796.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1791.10 | 1789.19 | 1796.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:00:00 | 1791.10 | 1789.19 | 1796.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 1800.80 | 1787.31 | 1792.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:00:00 | 1800.80 | 1787.31 | 1792.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 1810.50 | 1791.95 | 1794.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:45:00 | 1810.00 | 1791.95 | 1794.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 1795.50 | 1794.48 | 1795.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 11:00:00 | 1795.50 | 1794.48 | 1795.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 11:15:00 | 1803.40 | 1796.26 | 1795.95 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 1780.30 | 1793.75 | 1795.25 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 14:15:00 | 1801.40 | 1795.80 | 1795.65 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 10:15:00 | 1785.70 | 1794.81 | 1795.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 11:15:00 | 1777.00 | 1791.25 | 1793.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 10:15:00 | 1753.20 | 1753.00 | 1769.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 11:00:00 | 1753.20 | 1753.00 | 1769.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 1768.20 | 1753.50 | 1765.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:00:00 | 1768.20 | 1753.50 | 1765.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 1749.40 | 1752.68 | 1763.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 1739.50 | 1752.85 | 1762.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 11:15:00 | 1703.10 | 1695.71 | 1695.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 11:15:00 | 1703.10 | 1695.71 | 1695.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 12:15:00 | 1707.10 | 1697.98 | 1696.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-09 09:15:00 | 1750.50 | 1760.71 | 1750.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 1750.50 | 1760.71 | 1750.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1750.50 | 1760.71 | 1750.08 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 15:15:00 | 1734.00 | 1745.06 | 1746.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 10:15:00 | 1729.00 | 1742.24 | 1744.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 14:15:00 | 1742.00 | 1740.16 | 1742.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 14:15:00 | 1742.00 | 1740.16 | 1742.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 1742.00 | 1740.16 | 1742.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 15:00:00 | 1742.00 | 1740.16 | 1742.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 1745.00 | 1741.13 | 1742.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 1745.20 | 1741.13 | 1742.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1740.00 | 1740.90 | 1742.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 10:15:00 | 1733.10 | 1740.90 | 1742.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 12:00:00 | 1730.60 | 1737.50 | 1740.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 13:45:00 | 1735.00 | 1736.67 | 1739.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 10:15:00 | 1648.25 | 1679.64 | 1691.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 11:15:00 | 1646.44 | 1673.62 | 1687.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 11:15:00 | 1644.07 | 1673.62 | 1687.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 13:15:00 | 1651.20 | 1649.36 | 1663.63 | SL hit (close>ema200) qty=0.50 sl=1649.36 alert=retest2 |

### Cycle 41 — BUY (started 2025-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 13:15:00 | 1675.40 | 1662.76 | 1661.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 12:15:00 | 1688.30 | 1671.74 | 1666.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 09:15:00 | 1730.00 | 1756.28 | 1735.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 1730.00 | 1756.28 | 1735.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1730.00 | 1756.28 | 1735.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 1730.00 | 1756.28 | 1735.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 1735.70 | 1752.16 | 1735.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:15:00 | 1724.90 | 1752.16 | 1735.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 1712.40 | 1744.21 | 1733.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:00:00 | 1712.40 | 1744.21 | 1733.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 1719.90 | 1739.35 | 1732.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:30:00 | 1711.80 | 1739.35 | 1732.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 1724.00 | 1730.68 | 1729.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 09:15:00 | 1800.90 | 1730.68 | 1729.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 1765.50 | 1805.50 | 1807.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 1765.50 | 1805.50 | 1807.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 12:15:00 | 1740.30 | 1792.46 | 1801.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 1762.60 | 1759.95 | 1778.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 12:00:00 | 1762.60 | 1759.95 | 1778.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 1789.60 | 1765.88 | 1779.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:30:00 | 1792.70 | 1765.88 | 1779.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 1790.20 | 1770.75 | 1780.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 1794.40 | 1770.75 | 1780.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 1787.30 | 1776.32 | 1781.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 1773.20 | 1776.32 | 1781.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 1805.40 | 1783.05 | 1783.54 | SL hit (close>static) qty=1.00 sl=1793.60 alert=retest2 |

### Cycle 43 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 1799.20 | 1786.28 | 1784.96 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 11:15:00 | 1778.10 | 1785.38 | 1786.33 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 1796.40 | 1787.26 | 1786.89 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 15:15:00 | 1777.80 | 1785.91 | 1786.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 09:15:00 | 1752.90 | 1779.30 | 1783.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 14:15:00 | 1741.10 | 1735.67 | 1750.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-14 15:00:00 | 1741.10 | 1735.67 | 1750.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 1753.20 | 1742.02 | 1748.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 12:00:00 | 1753.20 | 1742.02 | 1748.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 1766.70 | 1746.96 | 1750.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 12:45:00 | 1771.50 | 1746.96 | 1750.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 13:15:00 | 1802.10 | 1757.99 | 1755.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 14:15:00 | 1819.20 | 1770.23 | 1761.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 09:15:00 | 1831.00 | 1837.43 | 1817.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-20 10:00:00 | 1831.00 | 1837.43 | 1817.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1820.00 | 1840.07 | 1829.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:15:00 | 1811.70 | 1840.07 | 1829.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 1810.20 | 1834.10 | 1827.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:00:00 | 1810.20 | 1834.10 | 1827.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 12:15:00 | 1805.40 | 1823.08 | 1823.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 11:15:00 | 1797.80 | 1809.04 | 1815.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 1791.50 | 1776.08 | 1787.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 1791.50 | 1776.08 | 1787.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1791.50 | 1776.08 | 1787.30 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 10:15:00 | 1801.00 | 1791.03 | 1790.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 11:15:00 | 1804.80 | 1793.79 | 1791.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 10:15:00 | 1802.60 | 1803.22 | 1798.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 11:15:00 | 1803.40 | 1803.22 | 1798.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 11:15:00 | 1803.70 | 1803.32 | 1798.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 13:15:00 | 1807.30 | 1803.26 | 1799.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 09:15:00 | 1890.90 | 1924.64 | 1927.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 1890.90 | 1924.64 | 1927.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 1857.50 | 1905.71 | 1915.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 09:15:00 | 1791.00 | 1790.59 | 1825.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 10:00:00 | 1791.00 | 1790.59 | 1825.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 1827.40 | 1799.09 | 1823.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 1827.40 | 1799.09 | 1823.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 1823.60 | 1803.99 | 1823.45 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 1856.70 | 1832.65 | 1830.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 12:15:00 | 1874.20 | 1840.96 | 1834.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 15:15:00 | 1878.00 | 1878.30 | 1864.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:15:00 | 1884.90 | 1878.30 | 1864.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 1907.90 | 1916.95 | 1906.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 1907.90 | 1916.95 | 1906.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 1915.10 | 1916.58 | 1907.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 1916.80 | 1913.77 | 1908.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 10:15:00 | 1920.00 | 1912.82 | 1908.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 12:15:00 | 1902.20 | 1911.61 | 1909.42 | SL hit (close<static) qty=1.00 sl=1903.70 alert=retest2 |

### Cycle 52 — SELL (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 14:15:00 | 1903.00 | 1907.09 | 1907.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 1864.60 | 1897.14 | 1902.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 09:15:00 | 1841.70 | 1838.75 | 1855.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-01 09:45:00 | 1842.30 | 1838.75 | 1855.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 1691.40 | 1689.77 | 1702.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:45:00 | 1690.00 | 1689.77 | 1702.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1626.60 | 1642.46 | 1653.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:15:00 | 1617.20 | 1638.63 | 1650.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:15:00 | 1620.70 | 1635.89 | 1648.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:30:00 | 1621.00 | 1630.13 | 1643.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 1617.50 | 1627.30 | 1639.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1639.30 | 1629.70 | 1639.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:45:00 | 1630.70 | 1629.70 | 1639.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 1661.00 | 1635.96 | 1641.71 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 1661.00 | 1635.96 | 1641.71 | SL hit (close>static) qty=1.00 sl=1659.80 alert=retest2 |

### Cycle 53 — BUY (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 12:15:00 | 1673.70 | 1649.28 | 1647.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 14:15:00 | 1682.40 | 1660.10 | 1652.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 1641.50 | 1659.72 | 1654.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 09:15:00 | 1641.50 | 1659.72 | 1654.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 1641.50 | 1659.72 | 1654.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:00:00 | 1641.50 | 1659.72 | 1654.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 1642.70 | 1656.32 | 1652.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 13:30:00 | 1661.60 | 1654.82 | 1652.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 14:15:00 | 1667.50 | 1654.82 | 1652.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 15:15:00 | 1666.60 | 1655.60 | 1653.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-21 10:15:00 | 1638.20 | 1652.44 | 1652.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 1638.20 | 1652.44 | 1652.65 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 15:15:00 | 1660.00 | 1651.93 | 1651.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 09:15:00 | 1693.60 | 1660.27 | 1655.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 10:15:00 | 1691.40 | 1693.20 | 1678.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 11:00:00 | 1691.40 | 1693.20 | 1678.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 1666.20 | 1687.80 | 1677.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 12:00:00 | 1666.20 | 1687.80 | 1677.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 1650.30 | 1680.30 | 1674.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:00:00 | 1650.30 | 1680.30 | 1674.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 1635.60 | 1671.36 | 1671.41 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 15:15:00 | 1712.00 | 1676.71 | 1673.66 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 09:15:00 | 1641.10 | 1669.59 | 1670.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 10:15:00 | 1627.20 | 1661.11 | 1666.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 1637.50 | 1636.92 | 1651.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 15:00:00 | 1637.50 | 1636.92 | 1651.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 1643.50 | 1631.74 | 1643.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:00:00 | 1643.50 | 1631.74 | 1643.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 1641.30 | 1633.65 | 1643.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:45:00 | 1651.30 | 1633.65 | 1643.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 13:15:00 | 1630.60 | 1633.04 | 1642.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 1614.70 | 1639.03 | 1641.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 13:15:00 | 1653.10 | 1631.30 | 1634.99 | SL hit (close>static) qty=1.00 sl=1643.30 alert=retest2 |

### Cycle 59 — BUY (started 2026-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 15:15:00 | 1648.60 | 1638.01 | 1637.60 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 1610.10 | 1634.32 | 1636.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 1594.30 | 1620.02 | 1627.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 12:15:00 | 1443.00 | 1438.07 | 1470.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-05 13:00:00 | 1443.00 | 1438.07 | 1470.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 1583.80 | 1467.21 | 1481.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:00:00 | 1583.80 | 1467.21 | 1481.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 1559.20 | 1485.61 | 1488.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 15:15:00 | 1540.00 | 1485.61 | 1488.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 15:15:00 | 1540.00 | 1496.49 | 1492.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 15:15:00 | 1540.00 | 1496.49 | 1492.95 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 11:15:00 | 1487.00 | 1512.88 | 1515.80 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 10:15:00 | 1545.90 | 1521.12 | 1517.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 11:15:00 | 1546.70 | 1526.23 | 1520.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 09:15:00 | 1533.30 | 1540.29 | 1531.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 09:15:00 | 1533.30 | 1540.29 | 1531.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1533.30 | 1540.29 | 1531.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:15:00 | 1558.70 | 1541.78 | 1533.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 14:45:00 | 1551.70 | 1548.30 | 1538.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 11:45:00 | 1551.20 | 1544.44 | 1539.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 15:15:00 | 1525.00 | 1535.46 | 1536.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 15:15:00 | 1525.00 | 1535.46 | 1536.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 1509.60 | 1530.29 | 1534.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 14:15:00 | 1496.10 | 1495.40 | 1507.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 15:00:00 | 1496.10 | 1495.40 | 1507.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 1507.00 | 1496.94 | 1505.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:45:00 | 1510.80 | 1496.94 | 1505.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 1508.00 | 1499.15 | 1505.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:30:00 | 1505.60 | 1499.15 | 1505.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 1499.00 | 1499.12 | 1505.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 12:15:00 | 1497.30 | 1499.12 | 1505.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 1493.50 | 1501.96 | 1504.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 10:00:00 | 1496.50 | 1500.86 | 1504.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:15:00 | 1495.90 | 1500.97 | 1503.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 1486.00 | 1497.98 | 1502.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 13:15:00 | 1481.60 | 1495.10 | 1500.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 13:45:00 | 1477.60 | 1491.84 | 1498.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 13:15:00 | 1520.80 | 1497.51 | 1496.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 1520.80 | 1497.51 | 1496.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 1546.20 | 1511.98 | 1504.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 13:15:00 | 1493.80 | 1519.09 | 1511.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 13:15:00 | 1493.80 | 1519.09 | 1511.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 1493.80 | 1519.09 | 1511.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:00:00 | 1493.80 | 1519.09 | 1511.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 1493.60 | 1513.99 | 1509.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:30:00 | 1492.00 | 1513.99 | 1509.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 1488.70 | 1506.20 | 1506.70 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 1525.10 | 1506.47 | 1504.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 11:15:00 | 1530.20 | 1511.21 | 1507.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 13:15:00 | 1513.70 | 1514.14 | 1509.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 13:15:00 | 1513.70 | 1514.14 | 1509.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 1513.70 | 1514.14 | 1509.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 14:00:00 | 1513.70 | 1514.14 | 1509.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 1514.70 | 1515.21 | 1510.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 10:30:00 | 1527.00 | 1517.40 | 1512.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 12:45:00 | 1523.70 | 1519.09 | 1514.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 13:45:00 | 1521.40 | 1519.82 | 1514.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 1492.60 | 1515.18 | 1514.13 | SL hit (close<static) qty=1.00 sl=1507.10 alert=retest2 |

### Cycle 68 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 10:15:00 | 1482.70 | 1508.69 | 1511.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 12:15:00 | 1478.80 | 1498.68 | 1506.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 15:15:00 | 1474.00 | 1469.98 | 1482.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 09:30:00 | 1473.50 | 1469.48 | 1480.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 1472.20 | 1468.78 | 1478.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 11:30:00 | 1476.50 | 1468.78 | 1478.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 12:15:00 | 1482.60 | 1471.54 | 1478.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 13:00:00 | 1482.60 | 1471.54 | 1478.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 13:15:00 | 1482.60 | 1473.75 | 1479.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 13:30:00 | 1482.00 | 1473.75 | 1479.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 1475.80 | 1475.87 | 1478.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 10:15:00 | 1466.90 | 1475.87 | 1478.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:45:00 | 1467.70 | 1471.87 | 1475.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 15:15:00 | 1466.40 | 1471.50 | 1475.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1393.56 | 1426.88 | 1445.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1394.32 | 1426.88 | 1445.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1393.08 | 1426.88 | 1445.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 11:15:00 | 1423.70 | 1420.84 | 1439.32 | SL hit (close>ema200) qty=0.50 sl=1420.84 alert=retest2 |

### Cycle 69 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 1464.90 | 1443.09 | 1442.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 13:15:00 | 1470.00 | 1452.04 | 1446.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 11:15:00 | 1467.60 | 1468.88 | 1458.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 12:00:00 | 1467.60 | 1468.88 | 1458.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 1460.70 | 1468.22 | 1460.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 1460.70 | 1468.22 | 1460.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 1456.50 | 1465.88 | 1460.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 1449.60 | 1465.88 | 1460.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1452.30 | 1463.16 | 1459.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:15:00 | 1457.70 | 1463.16 | 1459.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:45:00 | 1463.50 | 1461.78 | 1459.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 12:45:00 | 1457.90 | 1461.00 | 1459.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 10:15:00 | 1449.80 | 1457.93 | 1458.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 1449.80 | 1457.93 | 1458.71 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-03-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 14:15:00 | 1477.40 | 1458.13 | 1456.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 09:15:00 | 1497.00 | 1467.80 | 1461.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1509.50 | 1517.71 | 1500.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 09:45:00 | 1508.80 | 1517.71 | 1500.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 1507.60 | 1513.67 | 1501.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 12:00:00 | 1507.60 | 1513.67 | 1501.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 12:15:00 | 1497.30 | 1510.40 | 1501.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:00:00 | 1497.30 | 1510.40 | 1501.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 1468.20 | 1501.96 | 1498.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:45:00 | 1480.00 | 1501.96 | 1498.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 1505.50 | 1504.14 | 1500.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:30:00 | 1505.00 | 1504.14 | 1500.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 1497.20 | 1503.18 | 1500.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 1497.20 | 1503.18 | 1500.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 1504.00 | 1503.34 | 1501.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:15:00 | 1472.70 | 1503.34 | 1501.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 1456.30 | 1493.93 | 1497.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 1449.80 | 1485.11 | 1492.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 10:15:00 | 1454.60 | 1451.89 | 1467.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:00:00 | 1454.60 | 1451.89 | 1467.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 1466.60 | 1454.83 | 1467.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 1466.60 | 1454.83 | 1467.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1471.30 | 1458.13 | 1467.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 1478.40 | 1458.13 | 1467.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1471.20 | 1460.74 | 1468.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:45:00 | 1472.60 | 1460.74 | 1468.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 1461.30 | 1460.85 | 1467.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:30:00 | 1461.60 | 1460.85 | 1467.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 1460.20 | 1460.72 | 1466.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 1496.90 | 1460.72 | 1466.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1480.00 | 1464.58 | 1468.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 10:45:00 | 1474.30 | 1467.16 | 1468.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 11:15:00 | 1472.40 | 1467.16 | 1468.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 12:15:00 | 1483.00 | 1471.58 | 1470.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 1483.00 | 1471.58 | 1470.73 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 1419.70 | 1461.52 | 1466.53 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 1469.60 | 1452.06 | 1451.82 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 14:15:00 | 1431.40 | 1448.68 | 1450.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 1396.40 | 1435.39 | 1443.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 14:15:00 | 1429.00 | 1423.94 | 1433.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 15:00:00 | 1429.00 | 1423.94 | 1433.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 1425.10 | 1424.17 | 1432.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:15:00 | 1415.30 | 1424.17 | 1432.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1443.60 | 1428.06 | 1433.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:00:00 | 1443.60 | 1428.06 | 1433.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 1437.20 | 1429.89 | 1434.22 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 1463.70 | 1441.73 | 1438.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 1469.60 | 1447.30 | 1441.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 1442.20 | 1449.91 | 1444.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 1442.20 | 1449.91 | 1444.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 1442.20 | 1449.91 | 1444.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 12:30:00 | 1467.20 | 1448.52 | 1444.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 14:30:00 | 1460.40 | 1451.43 | 1446.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 15:00:00 | 1461.10 | 1451.43 | 1446.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 1492.20 | 1452.58 | 1447.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 1481.70 | 1488.67 | 1474.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 1481.70 | 1488.67 | 1474.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1465.70 | 1498.34 | 1493.51 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 11:15:00 | 1465.00 | 1486.53 | 1488.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 11:15:00 | 1465.00 | 1486.53 | 1488.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 12:15:00 | 1462.40 | 1481.71 | 1486.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 1480.20 | 1471.04 | 1478.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 1480.20 | 1471.04 | 1478.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1480.20 | 1471.04 | 1478.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 12:30:00 | 1471.60 | 1474.89 | 1478.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 13:30:00 | 1472.10 | 1474.75 | 1478.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 10:15:00 | 1506.20 | 1484.69 | 1482.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 10:15:00 | 1506.20 | 1484.69 | 1482.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 13:15:00 | 1528.60 | 1499.42 | 1490.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 15:15:00 | 1605.20 | 1606.70 | 1580.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 09:15:00 | 1596.40 | 1606.70 | 1580.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 1596.70 | 1604.70 | 1582.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 11:45:00 | 1622.30 | 1607.92 | 1587.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 13:15:00 | 1661.10 | 1670.51 | 1670.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 13:15:00 | 1661.10 | 1670.51 | 1670.75 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 1698.40 | 1675.73 | 1673.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 1711.00 | 1682.79 | 1676.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 1684.60 | 1689.76 | 1681.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 13:15:00 | 1684.60 | 1689.76 | 1681.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 1684.60 | 1689.76 | 1681.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 1684.60 | 1689.76 | 1681.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 1683.50 | 1688.51 | 1681.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:30:00 | 1686.00 | 1688.51 | 1681.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 1680.50 | 1686.91 | 1681.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 1672.50 | 1686.91 | 1681.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1654.50 | 1680.43 | 1679.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 1654.50 | 1680.43 | 1679.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 1650.00 | 1674.34 | 1676.69 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 11:15:00 | 1673.90 | 1670.87 | 1670.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 14:15:00 | 1679.40 | 1672.84 | 1671.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 09:15:00 | 1668.10 | 1673.52 | 1672.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 1668.10 | 1673.52 | 1672.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 1668.10 | 1673.52 | 1672.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:00:00 | 1668.10 | 1673.52 | 1672.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 1712.40 | 1681.29 | 1675.93 | EMA400 retest candle locked (from upside) |

### Cycle 84 — SELL (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 10:15:00 | 1645.50 | 1677.57 | 1678.91 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2026-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 14:15:00 | 1683.50 | 1679.24 | 1679.08 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 09:15:00 | 1662.80 | 1676.88 | 1678.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 10:15:00 | 1657.60 | 1673.02 | 1676.22 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-27 11:15:00 | 1793.90 | 2025-05-28 14:15:00 | 1746.30 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2025-05-27 14:15:00 | 1780.90 | 2025-05-28 14:15:00 | 1746.30 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-05-28 10:45:00 | 1781.60 | 2025-05-28 14:15:00 | 1746.30 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-05-28 12:15:00 | 1783.20 | 2025-05-28 14:15:00 | 1746.30 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-06-16 09:15:00 | 1869.90 | 2025-06-16 13:15:00 | 1909.10 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-06-23 09:15:00 | 1886.20 | 2025-06-23 11:15:00 | 1903.10 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-06-27 12:00:00 | 1841.50 | 2025-07-02 11:15:00 | 1845.40 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-07-07 11:15:00 | 1794.10 | 2025-07-07 14:15:00 | 1827.80 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-07-07 14:15:00 | 1802.20 | 2025-07-07 14:15:00 | 1827.80 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-07-31 11:45:00 | 1802.40 | 2025-08-01 09:15:00 | 1779.50 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-08-01 10:45:00 | 1798.50 | 2025-08-01 11:15:00 | 1780.80 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-08-01 11:15:00 | 1802.00 | 2025-08-01 11:15:00 | 1780.80 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-08-05 09:15:00 | 1764.10 | 2025-08-07 15:15:00 | 1786.30 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-08-18 09:15:00 | 1899.90 | 2025-08-25 10:15:00 | 1897.80 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-09-26 09:15:00 | 1739.50 | 2025-10-06 11:15:00 | 1703.10 | STOP_HIT | 1.00 | 2.09% |
| SELL | retest2 | 2025-10-13 10:15:00 | 1733.10 | 2025-10-17 10:15:00 | 1648.25 | PARTIAL | 0.50 | 4.90% |
| SELL | retest2 | 2025-10-13 12:00:00 | 1730.60 | 2025-10-17 11:15:00 | 1646.44 | PARTIAL | 0.50 | 4.86% |
| SELL | retest2 | 2025-10-13 13:45:00 | 1735.00 | 2025-10-17 11:15:00 | 1644.07 | PARTIAL | 0.50 | 5.24% |
| SELL | retest2 | 2025-10-13 10:15:00 | 1733.10 | 2025-10-20 13:15:00 | 1651.20 | STOP_HIT | 0.50 | 4.73% |
| SELL | retest2 | 2025-10-13 12:00:00 | 1730.60 | 2025-10-20 13:15:00 | 1651.20 | STOP_HIT | 0.50 | 4.59% |
| SELL | retest2 | 2025-10-13 13:45:00 | 1735.00 | 2025-10-20 13:15:00 | 1651.20 | STOP_HIT | 0.50 | 4.83% |
| BUY | retest2 | 2025-10-30 09:15:00 | 1800.90 | 2025-11-06 11:15:00 | 1765.50 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-11-10 09:15:00 | 1773.20 | 2025-11-10 10:15:00 | 1805.40 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-11-28 13:15:00 | 1807.30 | 2025-12-15 09:15:00 | 1890.90 | STOP_HIT | 1.00 | 4.63% |
| BUY | retest2 | 2025-12-29 09:15:00 | 1916.80 | 2025-12-29 12:15:00 | 1902.20 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-12-29 10:15:00 | 1920.00 | 2025-12-29 12:15:00 | 1902.20 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2026-01-16 11:15:00 | 1617.20 | 2026-01-19 10:15:00 | 1661.00 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2026-01-16 12:15:00 | 1620.70 | 2026-01-19 10:15:00 | 1661.00 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2026-01-16 13:30:00 | 1621.00 | 2026-01-19 10:15:00 | 1661.00 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2026-01-19 09:15:00 | 1617.50 | 2026-01-19 10:15:00 | 1661.00 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2026-01-20 13:30:00 | 1661.60 | 2026-01-21 10:15:00 | 1638.20 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2026-01-20 14:15:00 | 1667.50 | 2026-01-21 10:15:00 | 1638.20 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2026-01-20 15:15:00 | 1666.60 | 2026-01-21 10:15:00 | 1638.20 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-01-30 09:15:00 | 1614.70 | 2026-01-30 13:15:00 | 1653.10 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2026-02-05 15:15:00 | 1540.00 | 2026-02-05 15:15:00 | 1540.00 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2026-02-12 12:15:00 | 1558.70 | 2026-02-13 15:15:00 | 1525.00 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2026-02-12 14:45:00 | 1551.70 | 2026-02-13 15:15:00 | 1525.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2026-02-13 11:45:00 | 1551.20 | 2026-02-13 15:15:00 | 1525.00 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2026-02-18 12:15:00 | 1497.30 | 2026-02-20 13:15:00 | 1520.80 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2026-02-19 09:15:00 | 1493.50 | 2026-02-20 13:15:00 | 1520.80 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-02-19 10:00:00 | 1496.50 | 2026-02-20 13:15:00 | 1520.80 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-02-19 11:15:00 | 1495.90 | 2026-02-20 13:15:00 | 1520.80 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2026-02-19 13:15:00 | 1481.60 | 2026-02-20 13:15:00 | 1520.80 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2026-02-19 13:45:00 | 1477.60 | 2026-02-20 13:15:00 | 1520.80 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2026-02-26 10:30:00 | 1527.00 | 2026-02-27 09:15:00 | 1492.60 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2026-02-26 12:45:00 | 1523.70 | 2026-02-27 09:15:00 | 1492.60 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2026-02-26 13:45:00 | 1521.40 | 2026-02-27 09:15:00 | 1492.60 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2026-03-05 10:15:00 | 1466.90 | 2026-03-09 09:15:00 | 1393.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 13:45:00 | 1467.70 | 2026-03-09 09:15:00 | 1394.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 15:15:00 | 1466.40 | 2026-03-09 09:15:00 | 1393.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 10:15:00 | 1466.90 | 2026-03-09 11:15:00 | 1423.70 | STOP_HIT | 0.50 | 2.94% |
| SELL | retest2 | 2026-03-05 13:45:00 | 1467.70 | 2026-03-09 11:15:00 | 1423.70 | STOP_HIT | 0.50 | 3.00% |
| SELL | retest2 | 2026-03-05 15:15:00 | 1466.40 | 2026-03-09 11:15:00 | 1423.70 | STOP_HIT | 0.50 | 2.91% |
| BUY | retest2 | 2026-03-12 10:15:00 | 1457.70 | 2026-03-13 10:15:00 | 1449.80 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2026-03-12 11:45:00 | 1463.50 | 2026-03-13 10:15:00 | 1449.80 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2026-03-12 12:45:00 | 1457.90 | 2026-03-13 10:15:00 | 1449.80 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2026-03-25 10:45:00 | 1474.30 | 2026-03-25 12:15:00 | 1483.00 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2026-03-25 11:15:00 | 1472.40 | 2026-03-25 12:15:00 | 1483.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2026-04-07 12:30:00 | 1467.20 | 2026-04-13 11:15:00 | 1465.00 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2026-04-07 14:30:00 | 1460.40 | 2026-04-13 11:15:00 | 1465.00 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2026-04-07 15:00:00 | 1461.10 | 2026-04-13 11:15:00 | 1465.00 | STOP_HIT | 1.00 | 0.27% |
| BUY | retest2 | 2026-04-08 09:15:00 | 1492.20 | 2026-04-13 11:15:00 | 1465.00 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2026-04-15 12:30:00 | 1471.60 | 2026-04-16 10:15:00 | 1506.20 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2026-04-15 13:30:00 | 1472.10 | 2026-04-16 10:15:00 | 1506.20 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2026-04-21 11:45:00 | 1622.30 | 2026-04-28 13:15:00 | 1661.10 | STOP_HIT | 1.00 | 2.39% |
