# Adani Green Energy Ltd. (ADANIGREEN)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1350.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 133 |
| ALERT1 | 90 |
| ALERT2 | 90 |
| ALERT2_SKIP | 43 |
| ALERT3 | 249 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 94 |
| PARTIAL | 26 |
| TARGET_HIT | 11 |
| STOP_HIT | 89 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 126 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 60 / 66
- **Target hits / Stop hits / Partials:** 11 / 89 / 26
- **Avg / median % per leg:** 1.14% / -0.39%
- **Sum % (uncompounded):** 144.18%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 10 | 25.0% | 0 | 37 | 3 | -1.32% | -52.7% |
| BUY @ 2nd Alert (retest1) | 9 | 6 | 66.7% | 0 | 6 | 3 | 2.11% | 19.0% |
| BUY @ 3rd Alert (retest2) | 31 | 4 | 12.9% | 0 | 31 | 0 | -2.31% | -71.6% |
| SELL (all) | 86 | 50 | 58.1% | 11 | 52 | 23 | 2.29% | 196.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 86 | 50 | 58.1% | 11 | 52 | 23 | 2.29% | 196.9% |
| retest1 (combined) | 9 | 6 | 66.7% | 0 | 6 | 3 | 2.11% | 19.0% |
| retest2 (combined) | 117 | 54 | 46.2% | 11 | 83 | 23 | 1.07% | 125.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 12:15:00 | 1756.00 | 1719.55 | 1717.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 14:15:00 | 1785.70 | 1739.55 | 1727.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 09:15:00 | 1809.10 | 1813.16 | 1782.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 10:00:00 | 1809.10 | 1813.16 | 1782.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 1841.30 | 1833.70 | 1824.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 10:15:00 | 1854.50 | 1833.70 | 1824.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 14:15:00 | 1898.25 | 1903.30 | 1903.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 14:15:00 | 1898.25 | 1903.30 | 1903.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 15:15:00 | 1885.75 | 1899.79 | 1902.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 13:15:00 | 1885.80 | 1875.63 | 1882.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 13:15:00 | 1885.80 | 1875.63 | 1882.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 13:15:00 | 1885.80 | 1875.63 | 1882.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 14:00:00 | 1885.80 | 1875.63 | 1882.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 14:15:00 | 1875.50 | 1875.60 | 1882.10 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 11:15:00 | 1898.05 | 1886.90 | 1885.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 12:15:00 | 1902.55 | 1890.03 | 1887.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-31 14:15:00 | 1889.70 | 1901.43 | 1893.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 14:15:00 | 1889.70 | 1901.43 | 1893.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 1889.70 | 1901.43 | 1893.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 1889.70 | 1901.43 | 1893.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 1912.90 | 1903.72 | 1895.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 09:15:00 | 2046.45 | 1903.72 | 1895.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 09:45:00 | 1921.00 | 1986.87 | 1954.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 10:15:00 | 1687.65 | 1927.03 | 1930.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 1687.65 | 1927.03 | 1930.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 1667.85 | 1875.19 | 1906.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 1747.20 | 1739.32 | 1812.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 11:00:00 | 1747.20 | 1739.32 | 1812.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 1784.00 | 1757.91 | 1803.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:45:00 | 1824.85 | 1757.91 | 1803.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 1818.10 | 1769.95 | 1804.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 15:00:00 | 1818.10 | 1769.95 | 1804.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 1827.00 | 1781.36 | 1806.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:15:00 | 1888.55 | 1781.36 | 1806.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 1895.00 | 1823.26 | 1822.56 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 11:15:00 | 1856.70 | 1862.70 | 1863.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-12 12:15:00 | 1849.70 | 1860.10 | 1861.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-14 09:15:00 | 1815.70 | 1814.95 | 1830.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-14 09:45:00 | 1816.05 | 1814.95 | 1830.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 1811.15 | 1807.72 | 1818.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 09:15:00 | 1781.15 | 1811.71 | 1815.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 09:15:00 | 1800.85 | 1801.58 | 1802.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 14:15:00 | 1803.60 | 1801.40 | 1801.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 13:15:00 | 1812.30 | 1799.25 | 1798.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 13:15:00 | 1812.30 | 1799.25 | 1798.59 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 12:15:00 | 1793.00 | 1798.54 | 1799.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 13:15:00 | 1787.65 | 1796.36 | 1798.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 14:15:00 | 1799.95 | 1797.08 | 1798.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 14:15:00 | 1799.95 | 1797.08 | 1798.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 1799.95 | 1797.08 | 1798.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 12:00:00 | 1787.00 | 1792.74 | 1795.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 12:30:00 | 1786.35 | 1791.62 | 1794.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 14:15:00 | 1812.55 | 1786.61 | 1788.33 | SL hit (close>static) qty=1.00 sl=1801.40 alert=retest2 |

### Cycle 9 — BUY (started 2024-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 15:15:00 | 1808.00 | 1790.89 | 1790.11 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 09:15:00 | 1779.80 | 1788.91 | 1789.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 14:15:00 | 1772.05 | 1778.46 | 1782.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-02 15:15:00 | 1779.75 | 1778.71 | 1781.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 15:15:00 | 1779.75 | 1778.71 | 1781.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 15:15:00 | 1779.75 | 1778.71 | 1781.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 09:15:00 | 1768.25 | 1778.71 | 1781.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 1772.25 | 1777.42 | 1781.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 10:30:00 | 1760.70 | 1766.70 | 1771.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-09 10:15:00 | 1776.65 | 1760.68 | 1759.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2024-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 10:15:00 | 1776.65 | 1760.68 | 1759.91 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 1753.90 | 1760.75 | 1760.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 10:15:00 | 1743.05 | 1757.21 | 1759.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 14:15:00 | 1758.80 | 1755.26 | 1757.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 14:15:00 | 1758.80 | 1755.26 | 1757.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 1758.80 | 1755.26 | 1757.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 15:00:00 | 1758.80 | 1755.26 | 1757.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 1752.35 | 1754.68 | 1757.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:15:00 | 1750.85 | 1754.68 | 1757.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 1747.00 | 1753.14 | 1756.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:45:00 | 1753.00 | 1753.14 | 1756.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 1752.00 | 1752.41 | 1755.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:00:00 | 1752.00 | 1752.41 | 1755.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 1752.00 | 1750.38 | 1753.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:15:00 | 1747.45 | 1750.38 | 1753.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 1742.55 | 1748.82 | 1752.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 12:45:00 | 1739.90 | 1745.40 | 1749.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 13:45:00 | 1738.75 | 1743.49 | 1748.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 11:00:00 | 1739.50 | 1739.33 | 1744.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 13:00:00 | 1739.00 | 1739.28 | 1743.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 1741.00 | 1739.61 | 1743.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 15:00:00 | 1741.00 | 1739.61 | 1743.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 15:15:00 | 1737.20 | 1739.13 | 1742.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:15:00 | 1798.05 | 1739.13 | 1742.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-16 09:15:00 | 1784.50 | 1748.20 | 1746.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 09:15:00 | 1784.50 | 1748.20 | 1746.36 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 11:15:00 | 1739.45 | 1752.25 | 1752.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 1728.50 | 1743.72 | 1747.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 09:15:00 | 1766.85 | 1732.54 | 1734.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 09:15:00 | 1766.85 | 1732.54 | 1734.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 1766.85 | 1732.54 | 1734.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:30:00 | 1775.75 | 1732.54 | 1734.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2024-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 10:15:00 | 1788.00 | 1743.63 | 1739.31 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 15:15:00 | 1722.00 | 1736.80 | 1738.10 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 14:15:00 | 1821.15 | 1750.02 | 1741.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 15:15:00 | 1848.00 | 1769.61 | 1750.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 14:15:00 | 1807.80 | 1812.23 | 1784.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-26 15:00:00 | 1807.80 | 1812.23 | 1784.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 1830.00 | 1844.66 | 1828.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 15:00:00 | 1830.00 | 1844.66 | 1828.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 1830.00 | 1841.73 | 1828.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:30:00 | 1819.00 | 1839.46 | 1828.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 1846.55 | 1840.88 | 1830.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 12:45:00 | 1856.55 | 1845.28 | 1834.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:15:00 | 1854.60 | 1847.04 | 1837.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 10:15:00 | 1794.25 | 1858.94 | 1863.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 1794.25 | 1858.94 | 1863.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 12:15:00 | 1780.50 | 1833.18 | 1850.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 10:15:00 | 1774.60 | 1770.96 | 1794.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 10:45:00 | 1778.00 | 1770.96 | 1794.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 12:15:00 | 1781.15 | 1777.04 | 1784.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 14:00:00 | 1778.65 | 1777.36 | 1784.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 15:00:00 | 1775.10 | 1776.91 | 1783.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 14:15:00 | 1778.80 | 1780.42 | 1782.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 15:15:00 | 1774.00 | 1780.38 | 1782.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 1774.00 | 1779.11 | 1781.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:15:00 | 1722.10 | 1779.11 | 1781.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1689.72 | 1767.37 | 1776.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1686.34 | 1767.37 | 1776.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1689.86 | 1767.37 | 1776.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1685.30 | 1767.37 | 1776.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 11:00:00 | 1752.80 | 1764.46 | 1774.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 13:15:00 | 1786.70 | 1767.26 | 1772.78 | SL hit (close>ema200) qty=0.50 sl=1767.26 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 09:15:00 | 1825.35 | 1785.73 | 1780.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 1871.05 | 1824.66 | 1812.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 12:15:00 | 1913.45 | 1917.70 | 1893.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 13:00:00 | 1913.45 | 1917.70 | 1893.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 1909.35 | 1913.44 | 1900.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 09:45:00 | 1914.80 | 1900.66 | 1898.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 12:15:00 | 1899.05 | 1902.70 | 1899.91 | SL hit (close<static) qty=1.00 sl=1900.20 alert=retest2 |

### Cycle 20 — SELL (started 2024-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 10:15:00 | 1884.15 | 1896.62 | 1898.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 13:15:00 | 1877.90 | 1889.04 | 1893.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 14:15:00 | 1890.75 | 1889.38 | 1893.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-26 15:00:00 | 1890.75 | 1889.38 | 1893.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 1891.45 | 1889.57 | 1892.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 14:00:00 | 1882.30 | 1886.87 | 1890.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 09:15:00 | 1873.00 | 1886.21 | 1889.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 10:15:00 | 1882.10 | 1845.89 | 1847.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-02 10:15:00 | 1870.85 | 1850.88 | 1849.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 10:15:00 | 1870.85 | 1850.88 | 1849.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 12:15:00 | 1892.95 | 1861.86 | 1854.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 14:15:00 | 1897.50 | 1904.47 | 1889.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-03 15:00:00 | 1897.50 | 1904.47 | 1889.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 1892.90 | 1900.48 | 1889.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 12:45:00 | 1907.90 | 1902.12 | 1892.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 09:15:00 | 1920.40 | 1902.09 | 1900.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 09:15:00 | 1882.00 | 1898.07 | 1898.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 09:15:00 | 1882.00 | 1898.07 | 1898.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 1857.10 | 1885.66 | 1892.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 1875.75 | 1865.10 | 1874.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 1875.75 | 1865.10 | 1874.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 1875.75 | 1865.10 | 1874.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:30:00 | 1876.45 | 1865.10 | 1874.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 1886.00 | 1869.28 | 1875.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 11:00:00 | 1886.00 | 1869.28 | 1875.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 11:15:00 | 1877.65 | 1870.96 | 1875.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 15:00:00 | 1845.25 | 1870.12 | 1874.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-16 09:15:00 | 1892.60 | 1817.37 | 1822.92 | SL hit (close>static) qty=1.00 sl=1887.95 alert=retest2 |

### Cycle 23 — BUY (started 2024-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 10:15:00 | 1903.15 | 1834.52 | 1830.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 11:15:00 | 1918.70 | 1851.36 | 1838.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 09:15:00 | 1935.95 | 1938.31 | 1909.15 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 10:30:00 | 1959.15 | 1943.36 | 1914.09 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 11:00:00 | 1963.55 | 1943.36 | 1914.09 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 1912.15 | 1940.55 | 1926.44 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-19 09:15:00 | 1912.15 | 1940.55 | 1926.44 | SL hit (close<ema400) qty=1.00 sl=1926.44 alert=retest1 |

### Cycle 24 — SELL (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 14:15:00 | 1973.75 | 2048.12 | 2049.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 09:15:00 | 1957.60 | 2018.80 | 2035.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 1766.40 | 1765.65 | 1797.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 09:45:00 | 1772.25 | 1765.65 | 1797.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 1804.50 | 1777.91 | 1795.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:00:00 | 1804.50 | 1777.91 | 1795.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 1802.20 | 1782.77 | 1795.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:45:00 | 1807.75 | 1782.77 | 1795.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 1813.45 | 1793.89 | 1798.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 1816.05 | 1793.89 | 1798.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 1812.45 | 1803.38 | 1802.52 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 14:15:00 | 1783.55 | 1800.73 | 1801.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 10:15:00 | 1777.20 | 1792.39 | 1797.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-10 14:15:00 | 1782.10 | 1780.69 | 1789.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-10 15:00:00 | 1782.10 | 1780.69 | 1789.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 1787.85 | 1782.12 | 1789.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:15:00 | 1776.55 | 1782.12 | 1789.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 1788.00 | 1783.30 | 1789.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:45:00 | 1787.30 | 1783.30 | 1789.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 1772.80 | 1781.20 | 1787.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 11:15:00 | 1768.50 | 1781.20 | 1787.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-11 14:15:00 | 1791.50 | 1783.00 | 1786.35 | SL hit (close>static) qty=1.00 sl=1788.60 alert=retest2 |

### Cycle 27 — BUY (started 2024-11-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 15:15:00 | 1639.40 | 1624.99 | 1623.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 1654.70 | 1630.93 | 1626.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 1672.20 | 1682.83 | 1659.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 10:00:00 | 1672.20 | 1682.83 | 1659.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 1665.00 | 1674.24 | 1662.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:45:00 | 1663.95 | 1674.24 | 1662.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 1642.40 | 1667.87 | 1660.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:00:00 | 1642.40 | 1667.87 | 1660.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 1650.00 | 1664.30 | 1659.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 1632.15 | 1664.30 | 1659.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 1623.70 | 1650.94 | 1654.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 1615.00 | 1643.76 | 1650.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 1502.80 | 1498.75 | 1528.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 1502.80 | 1498.75 | 1528.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 14:15:00 | 1481.40 | 1495.25 | 1515.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 14:45:00 | 1511.85 | 1495.25 | 1515.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 1486.50 | 1493.14 | 1511.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 11:00:00 | 1470.10 | 1488.53 | 1507.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 13:15:00 | 1467.40 | 1484.27 | 1502.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 14:15:00 | 1462.95 | 1481.59 | 1499.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-21 09:15:00 | 1323.09 | 1381.79 | 1436.20 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 10:15:00 | 1087.20 | 1010.58 | 1010.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 09:15:00 | 1216.00 | 1092.87 | 1055.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 13:15:00 | 1316.00 | 1316.62 | 1270.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 09:15:00 | 1274.90 | 1306.95 | 1277.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 1274.90 | 1306.95 | 1277.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:30:00 | 1281.90 | 1306.95 | 1277.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 1264.80 | 1298.52 | 1276.10 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2024-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 09:15:00 | 1234.95 | 1267.50 | 1268.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 11:15:00 | 1226.70 | 1236.36 | 1247.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 11:15:00 | 1214.95 | 1211.93 | 1227.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-09 11:15:00 | 1214.95 | 1211.93 | 1227.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 11:15:00 | 1214.95 | 1211.93 | 1227.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 11:45:00 | 1216.05 | 1211.93 | 1227.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 13:15:00 | 1218.75 | 1214.91 | 1226.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 13:30:00 | 1221.30 | 1214.91 | 1226.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 1194.50 | 1211.88 | 1222.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 11:30:00 | 1188.00 | 1203.43 | 1216.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 09:15:00 | 1128.60 | 1152.59 | 1173.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-12 10:15:00 | 1244.50 | 1170.97 | 1180.14 | SL hit (close>ema200) qty=0.50 sl=1170.97 alert=retest2 |

### Cycle 31 — BUY (started 2024-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 12:15:00 | 1242.00 | 1193.99 | 1189.56 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 10:15:00 | 1174.35 | 1194.74 | 1196.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-16 11:15:00 | 1170.95 | 1189.98 | 1194.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 11:15:00 | 1088.05 | 1083.42 | 1102.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 11:15:00 | 1088.05 | 1083.42 | 1102.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 1088.05 | 1083.42 | 1102.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 11:45:00 | 1096.55 | 1083.42 | 1102.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 1038.50 | 1037.74 | 1055.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:30:00 | 1046.95 | 1037.74 | 1055.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 10:15:00 | 1042.50 | 1034.66 | 1044.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 11:00:00 | 1042.50 | 1034.66 | 1044.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 11:15:00 | 1035.75 | 1034.88 | 1043.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 11:30:00 | 1040.25 | 1034.88 | 1043.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 12:15:00 | 1069.00 | 1041.70 | 1045.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 13:00:00 | 1069.00 | 1041.70 | 1045.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 13:15:00 | 1071.90 | 1047.74 | 1048.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 13:45:00 | 1077.00 | 1047.74 | 1048.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2024-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 14:15:00 | 1063.35 | 1050.86 | 1049.70 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 10:15:00 | 1037.85 | 1053.87 | 1054.72 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 09:15:00 | 1055.85 | 1047.59 | 1047.46 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-01-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 14:15:00 | 1039.95 | 1046.51 | 1047.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 09:15:00 | 1017.00 | 1039.41 | 1043.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 1004.15 | 1002.35 | 1018.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 09:30:00 | 1012.15 | 1002.35 | 1018.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 973.35 | 925.52 | 942.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 10:00:00 | 973.35 | 925.52 | 942.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 1001.85 | 940.79 | 947.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 1001.85 | 940.79 | 947.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 11:15:00 | 1009.70 | 954.57 | 953.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 09:15:00 | 1041.45 | 997.25 | 977.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 11:15:00 | 1065.45 | 1067.49 | 1046.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 12:00:00 | 1065.45 | 1067.49 | 1046.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 1061.40 | 1069.33 | 1055.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 10:00:00 | 1061.40 | 1069.33 | 1055.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 1044.55 | 1063.49 | 1059.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 1044.55 | 1063.49 | 1059.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 1050.40 | 1060.87 | 1058.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 11:30:00 | 1054.65 | 1060.17 | 1058.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 12:30:00 | 1052.00 | 1058.77 | 1058.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 14:00:00 | 1054.50 | 1057.92 | 1057.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 14:15:00 | 1046.55 | 1055.65 | 1056.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 14:15:00 | 1046.55 | 1055.65 | 1056.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 1016.80 | 1046.02 | 1052.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 1032.75 | 1028.82 | 1039.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 1032.75 | 1028.82 | 1039.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 1032.75 | 1028.82 | 1039.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 1032.75 | 1028.82 | 1039.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1041.55 | 1031.72 | 1038.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 1041.55 | 1031.72 | 1038.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1033.70 | 1032.11 | 1038.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 11:30:00 | 1022.05 | 1030.84 | 1037.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 14:45:00 | 1024.30 | 1031.28 | 1035.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 15:15:00 | 1022.65 | 1031.28 | 1035.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 10:00:00 | 1017.75 | 1027.20 | 1033.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 1026.45 | 1027.05 | 1032.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:45:00 | 1024.00 | 1027.05 | 1032.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 11:15:00 | 1025.75 | 1026.79 | 1031.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 11:45:00 | 1028.10 | 1026.79 | 1031.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 971.55 | 996.15 | 1008.60 | EMA400 retest candle locked (from downside) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 970.95 | 996.15 | 1008.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 973.08 | 996.15 | 1008.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 971.52 | 996.15 | 1008.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 10:15:00 | 966.86 | 992.69 | 1005.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 12:15:00 | 1003.00 | 994.00 | 1004.15 | SL hit (close>ema200) qty=0.50 sl=994.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 14:15:00 | 1000.45 | 990.69 | 990.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 15:15:00 | 1006.00 | 993.75 | 991.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 995.90 | 1008.06 | 1000.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 995.90 | 1008.06 | 1000.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 995.90 | 1008.06 | 1000.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 995.90 | 1008.06 | 1000.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1007.95 | 1008.03 | 1001.50 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 10:15:00 | 982.30 | 996.08 | 997.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 13:15:00 | 971.20 | 985.58 | 991.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 983.00 | 980.62 | 987.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 983.00 | 980.62 | 987.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 983.00 | 980.62 | 987.62 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 10:15:00 | 1004.90 | 989.04 | 987.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 12:15:00 | 1008.80 | 994.59 | 990.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 11:15:00 | 1001.45 | 1004.49 | 998.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 12:00:00 | 1001.45 | 1004.49 | 998.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 1002.30 | 1004.05 | 999.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 1002.30 | 1004.05 | 999.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 996.90 | 1002.62 | 998.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:45:00 | 997.10 | 1002.62 | 998.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 997.90 | 1001.68 | 998.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 15:15:00 | 995.40 | 1001.68 | 998.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 995.40 | 1000.42 | 998.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:15:00 | 991.55 | 1000.42 | 998.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 996.90 | 999.72 | 998.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:45:00 | 989.85 | 999.72 | 998.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 1001.65 | 1000.10 | 998.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:15:00 | 1006.55 | 1000.10 | 998.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 13:15:00 | 989.45 | 996.90 | 997.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 13:15:00 | 989.45 | 996.90 | 997.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 971.30 | 989.78 | 993.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 09:15:00 | 972.85 | 966.49 | 977.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-11 09:15:00 | 972.85 | 966.49 | 977.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 972.85 | 966.49 | 977.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 09:30:00 | 983.25 | 966.49 | 977.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 10:15:00 | 970.00 | 967.19 | 976.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 10:30:00 | 971.80 | 967.19 | 976.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 936.80 | 928.09 | 942.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:30:00 | 922.45 | 927.90 | 938.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 13:15:00 | 876.33 | 895.16 | 914.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 12:15:00 | 875.05 | 872.98 | 893.22 | SL hit (close>ema200) qty=0.50 sl=872.98 alert=retest2 |

### Cycle 43 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 827.85 | 797.87 | 796.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 12:15:00 | 839.20 | 806.14 | 800.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 15:15:00 | 843.20 | 843.62 | 829.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 09:15:00 | 837.40 | 843.62 | 829.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 827.40 | 838.38 | 830.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:00:00 | 827.40 | 838.38 | 830.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 829.75 | 836.65 | 830.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 14:30:00 | 836.50 | 835.36 | 830.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:30:00 | 839.80 | 837.51 | 832.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 09:15:00 | 821.40 | 832.39 | 833.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 821.40 | 832.39 | 833.06 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 14:15:00 | 857.00 | 829.68 | 828.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 09:15:00 | 888.45 | 845.47 | 836.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 15:15:00 | 909.00 | 909.07 | 899.69 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 09:15:00 | 917.05 | 909.07 | 899.69 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 10:15:00 | 913.70 | 907.92 | 900.02 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 12:00:00 | 912.70 | 910.02 | 902.42 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 10:15:00 | 962.90 | 930.47 | 916.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 10:15:00 | 959.39 | 930.47 | 916.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 10:15:00 | 958.34 | 930.47 | 916.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-03-24 10:15:00 | 951.00 | 951.32 | 936.24 | SL hit (close<ema200) qty=0.50 sl=951.32 alert=retest1 |

### Cycle 46 — SELL (started 2025-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 14:15:00 | 926.00 | 938.58 | 939.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 15:15:00 | 918.00 | 934.46 | 937.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 09:15:00 | 937.40 | 935.05 | 937.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-26 09:15:00 | 937.40 | 935.05 | 937.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 937.40 | 935.05 | 937.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 10:00:00 | 937.40 | 935.05 | 937.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 933.10 | 934.66 | 937.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 11:30:00 | 926.55 | 932.44 | 936.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 12:00:00 | 928.55 | 926.59 | 930.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 13:15:00 | 951.30 | 933.77 | 932.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 13:15:00 | 951.30 | 933.77 | 932.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 14:15:00 | 958.50 | 938.71 | 935.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 13:15:00 | 952.00 | 953.29 | 945.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 13:15:00 | 952.00 | 953.29 | 945.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 952.00 | 953.29 | 945.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 14:00:00 | 952.00 | 953.29 | 945.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 948.40 | 952.32 | 945.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 15:00:00 | 948.40 | 952.32 | 945.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 941.05 | 949.37 | 945.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:00:00 | 941.05 | 949.37 | 945.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 930.25 | 945.55 | 944.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:00:00 | 930.25 | 945.55 | 944.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 11:15:00 | 931.50 | 942.74 | 943.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-02 09:15:00 | 916.35 | 928.91 | 935.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 10:15:00 | 937.40 | 930.60 | 935.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 10:15:00 | 937.40 | 930.60 | 935.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 937.40 | 930.60 | 935.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 11:00:00 | 937.40 | 930.60 | 935.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 938.10 | 932.10 | 935.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:00:00 | 938.10 | 932.10 | 935.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 938.95 | 933.47 | 936.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 13:30:00 | 933.35 | 933.80 | 936.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 09:45:00 | 931.70 | 937.17 | 937.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 10:15:00 | 951.40 | 940.02 | 938.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 10:15:00 | 951.40 | 940.02 | 938.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 11:15:00 | 959.15 | 943.85 | 940.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 928.50 | 946.68 | 944.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 928.50 | 946.68 | 944.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 928.50 | 946.68 | 944.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 928.50 | 946.68 | 944.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 939.00 | 945.14 | 943.56 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 925.75 | 939.31 | 941.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 924.00 | 936.25 | 939.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 890.50 | 884.38 | 900.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 11:00:00 | 890.50 | 884.38 | 900.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 892.85 | 869.83 | 878.03 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-04-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 13:15:00 | 886.30 | 882.96 | 882.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 15:15:00 | 894.45 | 886.78 | 884.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 14:15:00 | 947.60 | 949.29 | 937.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 15:00:00 | 947.60 | 949.29 | 937.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 13:15:00 | 950.95 | 954.79 | 950.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 13:45:00 | 949.15 | 954.79 | 950.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 14:15:00 | 944.00 | 952.63 | 949.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 14:45:00 | 944.60 | 952.63 | 949.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 944.25 | 950.95 | 949.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 951.80 | 950.95 | 949.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 930.65 | 946.89 | 947.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 09:15:00 | 930.65 | 946.89 | 947.75 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 15:15:00 | 951.00 | 946.78 | 946.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 09:15:00 | 961.10 | 949.64 | 948.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 12:15:00 | 950.00 | 951.14 | 949.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-24 12:15:00 | 950.00 | 951.14 | 949.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 950.00 | 951.14 | 949.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 12:45:00 | 952.20 | 951.14 | 949.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 966.00 | 954.11 | 950.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 15:00:00 | 969.80 | 957.25 | 952.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 924.50 | 952.96 | 951.55 | SL hit (close<static) qty=1.00 sl=949.50 alert=retest2 |

### Cycle 54 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 910.50 | 944.47 | 947.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 14:15:00 | 902.05 | 914.03 | 922.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 922.85 | 912.91 | 920.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 922.85 | 912.91 | 920.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 922.85 | 912.91 | 920.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:30:00 | 920.40 | 912.91 | 920.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 918.60 | 914.05 | 920.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:15:00 | 915.45 | 914.05 | 920.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 09:15:00 | 936.75 | 915.06 | 917.52 | SL hit (close>static) qty=1.00 sl=931.95 alert=retest2 |

### Cycle 55 — BUY (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 10:15:00 | 957.25 | 923.49 | 921.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 11:15:00 | 991.00 | 937.00 | 927.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 947.30 | 953.94 | 941.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 10:00:00 | 947.30 | 953.94 | 941.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 938.90 | 950.94 | 940.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:00:00 | 938.90 | 950.94 | 940.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 936.65 | 948.08 | 940.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:30:00 | 928.90 | 948.08 | 940.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2025-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 15:15:00 | 921.90 | 935.17 | 936.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 09:15:00 | 911.30 | 930.40 | 933.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-08 11:15:00 | 922.55 | 921.31 | 925.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 11:15:00 | 922.55 | 921.31 | 925.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 11:15:00 | 922.55 | 921.31 | 925.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 11:30:00 | 922.05 | 921.31 | 925.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 934.95 | 890.30 | 897.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:45:00 | 939.10 | 890.30 | 897.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 933.65 | 906.76 | 904.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 941.55 | 921.61 | 912.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 959.40 | 962.65 | 951.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 09:45:00 | 958.20 | 962.65 | 951.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1016.00 | 1014.07 | 1000.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 1013.35 | 1014.07 | 1000.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 1006.00 | 1012.27 | 1002.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 1004.95 | 1012.27 | 1002.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 1000.50 | 1009.92 | 1002.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:45:00 | 1000.30 | 1009.92 | 1002.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 990.90 | 1006.12 | 1001.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 990.90 | 1006.12 | 1001.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 995.05 | 998.40 | 998.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 995.05 | 998.40 | 998.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 986.00 | 995.92 | 997.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 13:15:00 | 981.35 | 989.65 | 992.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 10:15:00 | 990.20 | 988.49 | 991.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 11:00:00 | 990.20 | 988.49 | 991.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 991.00 | 988.99 | 991.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:00:00 | 991.00 | 988.99 | 991.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 992.10 | 989.62 | 991.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 13:30:00 | 987.45 | 989.05 | 990.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 09:15:00 | 1002.25 | 991.24 | 991.39 | SL hit (close>static) qty=1.00 sl=994.80 alert=retest2 |

### Cycle 59 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 1017.90 | 996.57 | 993.80 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 10:15:00 | 1006.00 | 1007.38 | 1007.42 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 11:15:00 | 1008.70 | 1007.65 | 1007.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 12:15:00 | 1021.85 | 1010.49 | 1008.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 1014.00 | 1014.34 | 1011.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 10:00:00 | 1014.00 | 1014.34 | 1011.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 1009.20 | 1013.31 | 1011.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:30:00 | 1009.75 | 1013.31 | 1011.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 1011.20 | 1012.89 | 1011.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 13:30:00 | 1017.70 | 1013.64 | 1011.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 14:45:00 | 1018.00 | 1013.56 | 1012.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 09:30:00 | 1017.50 | 1014.54 | 1012.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 12:00:00 | 1017.50 | 1015.12 | 1013.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 1010.70 | 1014.19 | 1013.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:45:00 | 1009.50 | 1014.19 | 1013.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 1009.50 | 1013.25 | 1012.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:15:00 | 1004.00 | 1013.25 | 1012.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-03 10:15:00 | 1001.50 | 1010.85 | 1011.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 10:15:00 | 1001.50 | 1010.85 | 1011.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 995.50 | 1006.51 | 1009.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 12:15:00 | 1003.80 | 999.04 | 1003.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 12:15:00 | 1003.80 | 999.04 | 1003.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 1003.80 | 999.04 | 1003.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:00:00 | 1003.80 | 999.04 | 1003.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 1005.60 | 1000.35 | 1003.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:00:00 | 1005.60 | 1000.35 | 1003.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 1002.90 | 1000.86 | 1003.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:15:00 | 1005.90 | 1000.86 | 1003.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 1005.90 | 1001.87 | 1003.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:15:00 | 1016.70 | 1001.87 | 1003.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 1014.90 | 1004.48 | 1004.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:00:00 | 1014.90 | 1004.48 | 1004.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 10:15:00 | 1014.50 | 1006.48 | 1005.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 12:15:00 | 1016.90 | 1010.56 | 1008.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 12:15:00 | 1047.70 | 1053.49 | 1044.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 13:00:00 | 1047.70 | 1053.49 | 1044.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 1039.60 | 1050.71 | 1043.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 1040.20 | 1050.71 | 1043.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 1047.70 | 1050.11 | 1044.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 15:15:00 | 1049.00 | 1050.11 | 1044.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 1033.70 | 1044.99 | 1043.10 | SL hit (close<static) qty=1.00 sl=1039.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 1036.90 | 1040.89 | 1041.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 1013.00 | 1035.31 | 1038.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 996.90 | 993.97 | 1005.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 996.90 | 993.97 | 1005.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 963.10 | 951.61 | 956.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:00:00 | 963.10 | 951.61 | 956.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 963.80 | 954.05 | 956.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:00:00 | 963.80 | 954.05 | 956.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 959.40 | 955.91 | 957.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 13:30:00 | 960.40 | 955.91 | 957.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 960.70 | 956.87 | 957.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:45:00 | 960.40 | 956.87 | 957.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 960.00 | 957.49 | 957.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:15:00 | 979.70 | 957.49 | 957.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 987.40 | 963.48 | 960.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 10:15:00 | 995.50 | 969.88 | 963.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 10:15:00 | 983.10 | 984.21 | 975.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 10:45:00 | 984.50 | 984.21 | 975.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 980.60 | 983.71 | 979.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:45:00 | 981.70 | 983.71 | 979.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 978.90 | 982.75 | 979.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:00:00 | 978.90 | 982.75 | 979.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 983.50 | 982.90 | 979.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 13:45:00 | 990.00 | 984.62 | 981.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 14:45:00 | 991.20 | 985.94 | 982.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 12:15:00 | 1011.10 | 1015.71 | 1016.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 12:15:00 | 1011.10 | 1015.71 | 1016.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 15:15:00 | 1007.30 | 1012.55 | 1014.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 992.50 | 990.31 | 994.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 15:00:00 | 992.50 | 990.31 | 994.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 998.00 | 992.39 | 995.13 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 09:15:00 | 1000.40 | 997.09 | 996.82 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 09:15:00 | 995.70 | 996.72 | 996.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 988.90 | 995.15 | 996.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 11:15:00 | 996.00 | 995.32 | 996.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 11:15:00 | 996.00 | 995.32 | 996.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 996.00 | 995.32 | 996.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:45:00 | 998.10 | 995.32 | 996.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 997.60 | 995.78 | 996.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:30:00 | 997.70 | 995.78 | 996.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 997.30 | 996.08 | 996.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:30:00 | 998.40 | 996.08 | 996.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 993.70 | 995.64 | 996.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:15:00 | 1006.00 | 995.64 | 996.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 09:15:00 | 1006.90 | 997.89 | 997.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 10:15:00 | 1021.60 | 1002.63 | 999.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 1040.00 | 1044.05 | 1037.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 10:00:00 | 1040.00 | 1044.05 | 1037.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 1039.50 | 1043.14 | 1038.06 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 15:15:00 | 1030.00 | 1035.42 | 1035.72 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 11:15:00 | 1037.60 | 1035.93 | 1035.87 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 09:15:00 | 1031.40 | 1035.44 | 1035.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 11:15:00 | 1027.30 | 1032.70 | 1034.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 1035.30 | 1026.36 | 1029.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 1035.30 | 1026.36 | 1029.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1035.30 | 1026.36 | 1029.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:30:00 | 1031.30 | 1026.36 | 1029.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 1030.50 | 1027.19 | 1029.98 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 15:15:00 | 1034.40 | 1031.29 | 1031.20 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 09:15:00 | 1026.80 | 1030.39 | 1030.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 1006.40 | 1021.16 | 1025.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 999.50 | 995.30 | 1007.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 10:00:00 | 999.50 | 995.30 | 1007.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 14:15:00 | 1006.40 | 997.43 | 1004.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 14:45:00 | 1004.70 | 997.43 | 1004.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 1009.00 | 999.75 | 1004.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:15:00 | 1017.70 | 999.75 | 1004.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 1008.10 | 1001.42 | 1004.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 10:15:00 | 996.80 | 1001.42 | 1004.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 10:45:00 | 1002.00 | 1001.41 | 1004.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 14:15:00 | 1014.20 | 1006.52 | 1006.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 1014.20 | 1006.52 | 1006.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 1018.80 | 1010.56 | 1008.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 14:15:00 | 1011.70 | 1013.53 | 1010.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-30 15:00:00 | 1011.70 | 1013.53 | 1010.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 1009.00 | 1012.63 | 1010.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 1002.40 | 1012.63 | 1010.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 997.00 | 1009.50 | 1009.43 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 999.20 | 1007.44 | 1008.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 981.90 | 999.07 | 1004.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 09:15:00 | 999.70 | 997.25 | 1002.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 999.70 | 997.25 | 1002.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 999.70 | 997.25 | 1002.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:30:00 | 1002.05 | 997.25 | 1002.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 987.75 | 984.14 | 990.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:30:00 | 988.35 | 984.14 | 990.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 988.85 | 985.08 | 990.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:45:00 | 989.65 | 985.08 | 990.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 988.70 | 985.80 | 990.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:30:00 | 988.50 | 985.80 | 990.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 991.50 | 986.94 | 990.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 09:15:00 | 987.15 | 986.94 | 990.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 09:15:00 | 937.79 | 958.65 | 969.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 938.40 | 938.09 | 953.62 | SL hit (close>ema200) qty=0.50 sl=938.09 alert=retest2 |

### Cycle 77 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 947.00 | 930.14 | 927.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 11:15:00 | 955.35 | 944.43 | 937.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 12:15:00 | 975.70 | 976.48 | 967.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 13:15:00 | 973.80 | 976.48 | 967.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 966.10 | 973.65 | 967.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 966.10 | 973.65 | 967.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 966.00 | 972.12 | 967.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 964.15 | 972.12 | 967.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 968.65 | 971.43 | 967.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 967.85 | 971.43 | 967.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 959.15 | 968.97 | 966.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 959.15 | 968.97 | 966.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 961.80 | 967.54 | 966.26 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 13:15:00 | 956.55 | 963.97 | 964.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 955.70 | 962.31 | 963.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 963.85 | 962.49 | 963.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 963.85 | 962.49 | 963.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 963.85 | 962.49 | 963.74 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2025-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 13:15:00 | 966.70 | 964.55 | 964.41 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 948.80 | 961.32 | 962.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 934.00 | 945.87 | 953.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 924.50 | 920.29 | 928.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 924.50 | 920.29 | 928.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 924.50 | 920.29 | 928.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 926.15 | 920.29 | 928.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 927.95 | 923.47 | 927.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 927.95 | 923.47 | 927.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 931.60 | 925.09 | 927.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 931.60 | 925.09 | 927.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 931.00 | 926.27 | 927.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 930.85 | 926.27 | 927.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 938.60 | 930.87 | 929.85 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 931.15 | 934.11 | 934.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 10:15:00 | 924.65 | 931.07 | 932.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 932.10 | 928.03 | 930.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 932.10 | 928.03 | 930.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 932.10 | 928.03 | 930.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 932.10 | 928.03 | 930.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 928.90 | 928.20 | 930.52 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 10:15:00 | 936.65 | 931.77 | 931.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 12:15:00 | 941.60 | 934.41 | 932.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 12:15:00 | 977.50 | 978.50 | 970.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 12:45:00 | 976.45 | 978.50 | 970.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 985.85 | 987.73 | 985.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:45:00 | 985.85 | 987.73 | 985.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 981.00 | 986.39 | 985.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 981.00 | 986.39 | 985.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 979.25 | 984.96 | 984.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:30:00 | 977.95 | 984.96 | 984.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2025-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 15:15:00 | 980.00 | 983.97 | 984.19 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 10:15:00 | 989.05 | 984.79 | 984.51 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 977.75 | 983.25 | 983.85 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 09:15:00 | 1008.45 | 986.70 | 985.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 12:15:00 | 1051.80 | 1005.86 | 994.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 09:15:00 | 1097.80 | 1125.05 | 1101.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 1097.80 | 1125.05 | 1101.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1097.80 | 1125.05 | 1101.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:45:00 | 1105.80 | 1125.05 | 1101.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 1108.95 | 1121.83 | 1101.83 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2025-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 13:15:00 | 1091.10 | 1098.38 | 1099.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 1077.95 | 1094.30 | 1097.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 15:15:00 | 1032.00 | 1028.23 | 1041.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-01 09:15:00 | 1048.10 | 1028.23 | 1041.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1040.80 | 1030.75 | 1041.35 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 1067.30 | 1048.22 | 1045.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 1073.40 | 1053.25 | 1048.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 1063.00 | 1064.57 | 1057.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 09:30:00 | 1064.20 | 1064.57 | 1057.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 1058.60 | 1063.38 | 1057.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 1059.60 | 1063.38 | 1057.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 1059.40 | 1062.58 | 1057.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:45:00 | 1058.00 | 1062.58 | 1057.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 1057.70 | 1061.60 | 1057.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:45:00 | 1059.50 | 1061.60 | 1057.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 1058.00 | 1060.88 | 1057.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 14:15:00 | 1058.20 | 1060.88 | 1057.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 1059.70 | 1060.65 | 1058.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:15:00 | 1086.90 | 1060.36 | 1058.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:30:00 | 1063.00 | 1064.81 | 1061.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:00:00 | 1064.30 | 1064.81 | 1061.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 1046.20 | 1059.41 | 1059.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 1046.20 | 1059.41 | 1059.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 1037.70 | 1055.07 | 1057.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 1049.80 | 1048.46 | 1052.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 1049.80 | 1048.46 | 1052.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1049.80 | 1048.46 | 1052.57 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 10:15:00 | 1060.00 | 1053.79 | 1053.13 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 12:15:00 | 1053.50 | 1054.98 | 1055.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 1041.00 | 1050.54 | 1052.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1041.90 | 1037.98 | 1043.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 1041.90 | 1037.98 | 1043.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1041.90 | 1037.98 | 1043.54 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 1058.40 | 1045.04 | 1044.50 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 1040.40 | 1048.54 | 1049.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 09:15:00 | 1030.10 | 1041.34 | 1045.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 1041.60 | 1036.84 | 1040.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 1041.60 | 1036.84 | 1040.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 1041.60 | 1036.84 | 1040.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 1038.60 | 1036.84 | 1040.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 1036.30 | 1036.73 | 1039.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:15:00 | 1044.90 | 1036.73 | 1039.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1051.30 | 1039.65 | 1040.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:15:00 | 1057.40 | 1039.65 | 1040.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 1053.80 | 1042.48 | 1042.15 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 1036.30 | 1043.04 | 1043.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 12:15:00 | 1032.50 | 1039.60 | 1041.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 1050.70 | 1018.42 | 1021.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 1050.70 | 1018.42 | 1021.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1050.70 | 1018.42 | 1021.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 1050.70 | 1018.42 | 1021.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 1104.60 | 1035.65 | 1029.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 12:15:00 | 1115.30 | 1061.91 | 1043.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 15:15:00 | 1136.00 | 1138.14 | 1117.06 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 09:15:00 | 1157.30 | 1138.14 | 1117.06 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 1122.60 | 1133.85 | 1123.19 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-03 13:15:00 | 1122.60 | 1133.85 | 1123.19 | SL hit (close<ema400) qty=1.00 sl=1123.19 alert=retest1 |

### Cycle 98 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 1094.50 | 1113.92 | 1116.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 13:15:00 | 1086.20 | 1102.90 | 1109.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 14:15:00 | 1062.90 | 1057.14 | 1070.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 14:15:00 | 1062.90 | 1057.14 | 1070.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 1062.90 | 1057.14 | 1070.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:30:00 | 1069.20 | 1057.14 | 1070.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1069.10 | 1059.83 | 1069.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 12:45:00 | 1064.10 | 1062.76 | 1068.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 13:30:00 | 1061.10 | 1062.47 | 1067.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 1088.30 | 1053.17 | 1055.45 | SL hit (close>static) qty=1.00 sl=1088.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 1076.10 | 1057.76 | 1057.32 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 10:15:00 | 1055.00 | 1074.39 | 1076.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 1048.90 | 1064.12 | 1069.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 14:15:00 | 1013.50 | 1011.63 | 1024.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 15:00:00 | 1013.50 | 1011.63 | 1024.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1028.80 | 1015.12 | 1023.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 1028.80 | 1015.12 | 1023.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1031.80 | 1018.46 | 1024.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 1031.80 | 1018.46 | 1024.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 1035.50 | 1027.13 | 1027.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 09:15:00 | 1040.60 | 1032.27 | 1029.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 10:15:00 | 1036.20 | 1040.87 | 1036.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 10:15:00 | 1036.20 | 1040.87 | 1036.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 1036.20 | 1040.87 | 1036.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:00:00 | 1036.20 | 1040.87 | 1036.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 1033.60 | 1039.41 | 1036.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:00:00 | 1033.60 | 1039.41 | 1036.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 1043.50 | 1040.23 | 1037.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:30:00 | 1030.80 | 1040.23 | 1037.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 1039.00 | 1039.98 | 1037.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:30:00 | 1040.50 | 1039.98 | 1037.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 1039.60 | 1039.91 | 1037.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:45:00 | 1036.20 | 1039.91 | 1037.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 1042.30 | 1040.39 | 1038.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 1035.10 | 1040.39 | 1038.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1028.40 | 1037.99 | 1037.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 1028.40 | 1037.99 | 1037.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 1030.50 | 1036.49 | 1036.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 13:15:00 | 1021.70 | 1030.08 | 1033.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 1020.70 | 1018.17 | 1023.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 10:00:00 | 1020.70 | 1018.17 | 1023.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1018.50 | 1018.23 | 1022.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:30:00 | 1021.40 | 1018.23 | 1022.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 1018.20 | 1013.82 | 1017.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:00:00 | 1018.20 | 1013.82 | 1017.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 1016.30 | 1014.32 | 1016.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:15:00 | 1018.10 | 1014.32 | 1016.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 1018.10 | 1015.07 | 1017.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 1013.60 | 1015.07 | 1017.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:00:00 | 1013.90 | 1005.77 | 1006.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:30:00 | 1014.20 | 1006.15 | 1006.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 1013.40 | 1004.59 | 1004.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 13:15:00 | 1013.40 | 1004.59 | 1004.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 15:15:00 | 1024.00 | 1010.75 | 1007.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 15:15:00 | 1043.90 | 1044.34 | 1034.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:15:00 | 1034.60 | 1044.34 | 1034.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1030.80 | 1041.63 | 1034.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 1031.70 | 1041.63 | 1034.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1032.50 | 1039.81 | 1034.13 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 1024.00 | 1030.61 | 1031.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 1021.60 | 1028.81 | 1030.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 11:15:00 | 1028.30 | 1027.84 | 1029.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 11:15:00 | 1028.30 | 1027.84 | 1029.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 1028.30 | 1027.84 | 1029.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:45:00 | 1029.40 | 1027.84 | 1029.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 1022.10 | 1026.69 | 1028.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 13:30:00 | 1021.00 | 1025.44 | 1028.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 14:00:00 | 1020.40 | 1025.44 | 1028.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 15:15:00 | 1018.00 | 1014.76 | 1017.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 1022.00 | 1019.17 | 1018.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 1022.00 | 1019.17 | 1018.80 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 15:15:00 | 1017.00 | 1019.18 | 1019.40 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 10:15:00 | 1021.30 | 1019.69 | 1019.60 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 1017.20 | 1019.29 | 1019.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 1014.90 | 1018.41 | 1019.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 09:15:00 | 1022.50 | 1018.55 | 1018.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 09:15:00 | 1022.50 | 1018.55 | 1018.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1022.50 | 1018.55 | 1018.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:30:00 | 1021.10 | 1018.55 | 1018.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 10:15:00 | 1022.60 | 1019.36 | 1019.28 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 15:15:00 | 1014.40 | 1019.29 | 1019.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 1011.90 | 1017.81 | 1018.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 09:15:00 | 1007.40 | 1007.39 | 1011.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 10:00:00 | 1007.40 | 1007.39 | 1011.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 1002.00 | 1006.31 | 1011.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 11:15:00 | 1001.50 | 1006.31 | 1011.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 13:15:00 | 1001.70 | 1004.37 | 1009.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 1013.60 | 1007.99 | 1009.56 | SL hit (close>static) qty=1.00 sl=1011.60 alert=retest2 |

### Cycle 111 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 1016.80 | 1010.95 | 1010.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 09:15:00 | 1033.20 | 1017.32 | 1014.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 10:15:00 | 1033.60 | 1034.17 | 1028.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 10:30:00 | 1031.70 | 1034.17 | 1028.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 1030.00 | 1033.33 | 1028.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:45:00 | 1029.50 | 1033.33 | 1028.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 1031.00 | 1033.24 | 1029.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 1031.00 | 1033.24 | 1029.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 1031.30 | 1032.86 | 1029.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:30:00 | 1030.20 | 1032.86 | 1029.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 1028.70 | 1032.02 | 1029.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 1024.80 | 1032.02 | 1029.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1031.30 | 1031.88 | 1029.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:30:00 | 1024.00 | 1031.88 | 1029.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 1027.40 | 1030.98 | 1029.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:00:00 | 1027.40 | 1030.98 | 1029.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 1028.10 | 1030.41 | 1029.52 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2026-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 13:15:00 | 1015.80 | 1026.52 | 1027.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 1012.30 | 1018.67 | 1021.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 936.80 | 936.72 | 947.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-14 10:00:00 | 936.80 | 936.72 | 947.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 936.70 | 937.37 | 942.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:30:00 | 941.30 | 937.37 | 942.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 907.80 | 917.38 | 926.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 12:00:00 | 898.80 | 911.19 | 921.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 12:30:00 | 895.90 | 908.07 | 919.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:45:00 | 900.60 | 890.79 | 897.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:30:00 | 897.30 | 892.91 | 897.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 896.50 | 893.63 | 897.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:15:00 | 892.50 | 899.18 | 899.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 12:15:00 | 853.86 | 875.78 | 887.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 12:15:00 | 851.10 | 875.78 | 887.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 12:15:00 | 855.57 | 875.78 | 887.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 12:15:00 | 852.43 | 875.78 | 887.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 12:15:00 | 847.88 | 875.78 | 887.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-23 13:15:00 | 808.92 | 858.97 | 879.02 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 113 — BUY (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 13:15:00 | 858.00 | 827.56 | 826.07 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 806.55 | 835.45 | 838.97 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 926.40 | 854.06 | 844.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 11:15:00 | 966.60 | 930.01 | 898.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 957.80 | 958.71 | 940.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 09:45:00 | 960.80 | 958.71 | 940.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 978.95 | 977.21 | 967.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:30:00 | 969.05 | 977.21 | 967.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 982.50 | 987.10 | 980.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:45:00 | 985.00 | 987.10 | 980.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 982.60 | 986.20 | 981.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:45:00 | 982.00 | 986.20 | 981.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 988.50 | 986.66 | 981.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:30:00 | 989.80 | 987.17 | 982.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:15:00 | 990.55 | 987.17 | 982.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 950.80 | 983.45 | 982.70 | SL hit (close<static) qty=1.00 sl=980.65 alert=retest2 |

### Cycle 116 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 954.95 | 977.75 | 980.18 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2026-02-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 13:15:00 | 992.45 | 975.47 | 973.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 1008.25 | 989.25 | 982.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 998.10 | 1000.68 | 990.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-18 10:00:00 | 998.10 | 1000.68 | 990.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 994.00 | 999.35 | 991.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:45:00 | 992.50 | 999.35 | 991.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 993.80 | 998.10 | 994.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 993.80 | 998.10 | 994.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 987.15 | 995.91 | 993.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:45:00 | 987.70 | 995.91 | 993.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 986.50 | 994.03 | 992.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 12:15:00 | 991.15 | 994.03 | 992.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 12:45:00 | 990.30 | 993.52 | 992.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 13:15:00 | 984.50 | 991.72 | 991.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 13:15:00 | 984.50 | 991.72 | 991.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 976.05 | 988.58 | 990.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 982.15 | 975.25 | 980.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 982.15 | 975.25 | 980.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 982.15 | 975.25 | 980.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:45:00 | 981.25 | 975.25 | 980.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 978.45 | 975.89 | 980.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:15:00 | 975.40 | 975.89 | 980.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 09:30:00 | 974.65 | 973.30 | 973.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 10:30:00 | 974.10 | 972.85 | 973.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 926.63 | 945.18 | 955.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 925.92 | 945.18 | 955.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 925.39 | 945.18 | 955.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 877.86 | 905.57 | 926.52 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 119 — BUY (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 11:15:00 | 869.35 | 859.15 | 859.10 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 851.00 | 858.97 | 859.86 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2026-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 14:15:00 | 862.80 | 860.31 | 860.25 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 10:15:00 | 854.55 | 860.66 | 860.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-17 11:15:00 | 851.15 | 858.76 | 860.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 13:15:00 | 864.50 | 859.55 | 860.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 13:15:00 | 864.50 | 859.55 | 860.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 864.50 | 859.55 | 860.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 864.50 | 859.55 | 860.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 869.90 | 861.62 | 861.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 877.70 | 866.01 | 863.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 869.45 | 879.82 | 873.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 869.45 | 879.82 | 873.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 869.45 | 879.82 | 873.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 869.45 | 879.82 | 873.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 868.60 | 877.58 | 872.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:15:00 | 867.30 | 877.58 | 872.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 869.25 | 875.91 | 872.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:30:00 | 868.25 | 875.91 | 872.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 858.65 | 870.23 | 870.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 852.40 | 866.67 | 868.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 882.95 | 868.85 | 869.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 882.95 | 868.85 | 869.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 882.95 | 868.85 | 869.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 885.30 | 868.85 | 869.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 873.20 | 869.72 | 869.70 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 11:15:00 | 866.35 | 869.05 | 869.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 14:15:00 | 863.80 | 867.69 | 868.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 824.85 | 822.06 | 836.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 824.85 | 822.06 | 836.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 835.65 | 824.78 | 836.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 835.40 | 824.78 | 836.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 837.40 | 827.30 | 836.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 840.00 | 827.30 | 836.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 838.00 | 829.44 | 836.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:45:00 | 840.15 | 829.44 | 836.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 839.00 | 831.35 | 837.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 863.00 | 831.35 | 837.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 864.80 | 843.24 | 841.89 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 832.55 | 843.06 | 844.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 818.45 | 832.30 | 837.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 842.80 | 823.72 | 829.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 842.80 | 823.72 | 829.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 842.80 | 823.72 | 829.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 842.80 | 823.72 | 829.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 843.15 | 827.61 | 831.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:45:00 | 845.45 | 827.61 | 831.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 853.95 | 837.21 | 835.12 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 11:15:00 | 822.50 | 834.75 | 835.72 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 859.85 | 838.51 | 836.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 09:15:00 | 892.00 | 851.69 | 843.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 12:15:00 | 1115.00 | 1115.19 | 1102.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-17 13:00:00 | 1115.00 | 1115.19 | 1102.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 1191.35 | 1207.99 | 1191.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:30:00 | 1180.35 | 1207.99 | 1191.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 1168.40 | 1200.08 | 1189.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:00:00 | 1168.40 | 1200.08 | 1189.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 1160.35 | 1192.13 | 1186.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:45:00 | 1162.30 | 1192.13 | 1186.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 1240.00 | 1245.01 | 1237.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 1234.40 | 1245.01 | 1237.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1211.35 | 1238.28 | 1234.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 1211.35 | 1238.28 | 1234.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 1207.95 | 1232.21 | 1232.42 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 1259.70 | 1232.80 | 1231.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 10:15:00 | 1290.80 | 1244.40 | 1236.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 12:15:00 | 1354.20 | 1361.01 | 1346.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 12:45:00 | 1356.80 | 1361.01 | 1346.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-14 09:30:00 | 1693.35 | 2024-05-14 11:15:00 | 1745.00 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2024-05-21 10:15:00 | 1854.50 | 2024-05-28 14:15:00 | 1898.25 | STOP_HIT | 1.00 | 2.36% |
| BUY | retest2 | 2024-06-03 09:15:00 | 2046.45 | 2024-06-04 10:15:00 | 1687.65 | STOP_HIT | 1.00 | -17.53% |
| BUY | retest2 | 2024-06-04 09:45:00 | 1921.00 | 2024-06-04 10:15:00 | 1687.65 | STOP_HIT | 1.00 | -12.15% |
| SELL | retest2 | 2024-06-19 09:15:00 | 1781.15 | 2024-06-24 13:15:00 | 1812.30 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-06-21 09:15:00 | 1800.85 | 2024-06-24 13:15:00 | 1812.30 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2024-06-21 14:15:00 | 1803.60 | 2024-06-24 13:15:00 | 1812.30 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2024-06-26 12:00:00 | 1787.00 | 2024-06-27 14:15:00 | 1812.55 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-06-26 12:30:00 | 1786.35 | 2024-06-27 14:15:00 | 1812.55 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-07-05 10:30:00 | 1760.70 | 2024-07-09 10:15:00 | 1776.65 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-07-12 12:45:00 | 1739.90 | 2024-07-16 09:15:00 | 1784.50 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2024-07-12 13:45:00 | 1738.75 | 2024-07-16 09:15:00 | 1784.50 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2024-07-15 11:00:00 | 1739.50 | 2024-07-16 09:15:00 | 1784.50 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2024-07-15 13:00:00 | 1739.00 | 2024-07-16 09:15:00 | 1784.50 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2024-07-31 12:45:00 | 1856.55 | 2024-08-05 10:15:00 | 1794.25 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest2 | 2024-08-01 09:15:00 | 1854.60 | 2024-08-05 10:15:00 | 1794.25 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2024-08-08 14:00:00 | 1778.65 | 2024-08-12 09:15:00 | 1689.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-08 15:00:00 | 1775.10 | 2024-08-12 09:15:00 | 1686.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-09 14:15:00 | 1778.80 | 2024-08-12 09:15:00 | 1689.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-09 15:15:00 | 1774.00 | 2024-08-12 09:15:00 | 1685.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-08 14:00:00 | 1778.65 | 2024-08-12 13:15:00 | 1786.70 | STOP_HIT | 0.50 | -0.45% |
| SELL | retest2 | 2024-08-08 15:00:00 | 1775.10 | 2024-08-12 13:15:00 | 1786.70 | STOP_HIT | 0.50 | -0.65% |
| SELL | retest2 | 2024-08-09 14:15:00 | 1778.80 | 2024-08-12 13:15:00 | 1786.70 | STOP_HIT | 0.50 | -0.44% |
| SELL | retest2 | 2024-08-09 15:15:00 | 1774.00 | 2024-08-12 13:15:00 | 1786.70 | STOP_HIT | 0.50 | -0.72% |
| SELL | retest2 | 2024-08-12 09:15:00 | 1722.10 | 2024-08-12 13:15:00 | 1786.70 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest2 | 2024-08-12 11:00:00 | 1752.80 | 2024-08-12 13:15:00 | 1786.70 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-08-23 09:45:00 | 1914.80 | 2024-08-23 12:15:00 | 1899.05 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-08-27 14:00:00 | 1882.30 | 2024-09-02 10:15:00 | 1870.85 | STOP_HIT | 1.00 | 0.61% |
| SELL | retest2 | 2024-08-28 09:15:00 | 1873.00 | 2024-09-02 10:15:00 | 1870.85 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2024-09-02 10:15:00 | 1882.10 | 2024-09-02 10:15:00 | 1870.85 | STOP_HIT | 1.00 | 0.60% |
| BUY | retest2 | 2024-09-04 12:45:00 | 1907.90 | 2024-09-06 09:15:00 | 1882.00 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-09-06 09:15:00 | 1920.40 | 2024-09-06 09:15:00 | 1882.00 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2024-09-10 15:00:00 | 1845.25 | 2024-09-16 09:15:00 | 1892.60 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest1 | 2024-09-18 10:30:00 | 1959.15 | 2024-09-19 09:15:00 | 1912.15 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest1 | 2024-09-18 11:00:00 | 1963.55 | 2024-09-19 09:15:00 | 1912.15 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2024-09-19 13:15:00 | 1953.90 | 2024-09-27 14:15:00 | 1973.75 | STOP_HIT | 1.00 | 1.02% |
| SELL | retest2 | 2024-10-11 11:15:00 | 1768.50 | 2024-10-11 14:15:00 | 1791.50 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-10-14 15:15:00 | 1771.00 | 2024-10-22 13:15:00 | 1682.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 09:45:00 | 1769.40 | 2024-10-22 13:15:00 | 1680.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 11:15:00 | 1768.75 | 2024-10-22 13:15:00 | 1680.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 11:45:00 | 1732.00 | 2024-10-23 09:15:00 | 1645.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-14 15:15:00 | 1771.00 | 2024-10-23 13:15:00 | 1686.00 | STOP_HIT | 0.50 | 4.80% |
| SELL | retest2 | 2024-10-15 09:45:00 | 1769.40 | 2024-10-23 13:15:00 | 1686.00 | STOP_HIT | 0.50 | 4.71% |
| SELL | retest2 | 2024-10-15 11:15:00 | 1768.75 | 2024-10-23 13:15:00 | 1686.00 | STOP_HIT | 0.50 | 4.68% |
| SELL | retest2 | 2024-10-21 11:45:00 | 1732.00 | 2024-10-23 13:15:00 | 1686.00 | STOP_HIT | 0.50 | 2.66% |
| SELL | retest2 | 2024-11-18 11:00:00 | 1470.10 | 2024-11-21 09:15:00 | 1323.09 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-18 13:15:00 | 1467.40 | 2024-11-21 09:15:00 | 1320.66 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-18 14:15:00 | 1462.95 | 2024-11-21 09:15:00 | 1316.65 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-10 11:30:00 | 1188.00 | 2024-12-12 09:15:00 | 1128.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-10 11:30:00 | 1188.00 | 2024-12-12 10:15:00 | 1244.50 | STOP_HIT | 0.50 | -4.76% |
| BUY | retest2 | 2025-01-21 11:30:00 | 1054.65 | 2025-01-21 14:15:00 | 1046.55 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-01-21 12:30:00 | 1052.00 | 2025-01-21 14:15:00 | 1046.55 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-01-21 14:00:00 | 1054.50 | 2025-01-21 14:15:00 | 1046.55 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-01-23 11:30:00 | 1022.05 | 2025-01-28 09:15:00 | 970.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 14:45:00 | 1024.30 | 2025-01-28 09:15:00 | 973.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 15:15:00 | 1022.65 | 2025-01-28 09:15:00 | 971.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 10:00:00 | 1017.75 | 2025-01-28 10:15:00 | 966.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 11:30:00 | 1022.05 | 2025-01-28 12:15:00 | 1003.00 | STOP_HIT | 0.50 | 1.86% |
| SELL | retest2 | 2025-01-23 14:45:00 | 1024.30 | 2025-01-28 12:15:00 | 1003.00 | STOP_HIT | 0.50 | 2.08% |
| SELL | retest2 | 2025-01-23 15:15:00 | 1022.65 | 2025-01-28 12:15:00 | 1003.00 | STOP_HIT | 0.50 | 1.92% |
| SELL | retest2 | 2025-01-24 10:00:00 | 1017.75 | 2025-01-28 12:15:00 | 1003.00 | STOP_HIT | 0.50 | 1.45% |
| SELL | retest2 | 2025-01-30 14:15:00 | 968.25 | 2025-01-31 14:15:00 | 1000.45 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest2 | 2025-02-07 11:15:00 | 1006.55 | 2025-02-07 13:15:00 | 989.45 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-02-13 13:30:00 | 922.45 | 2025-02-14 13:15:00 | 876.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:30:00 | 922.45 | 2025-02-17 12:15:00 | 875.05 | STOP_HIT | 0.50 | 5.14% |
| BUY | retest2 | 2025-03-07 14:30:00 | 836.50 | 2025-03-11 09:15:00 | 821.40 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-03-10 09:30:00 | 839.80 | 2025-03-11 09:15:00 | 821.40 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest1 | 2025-03-20 09:15:00 | 917.05 | 2025-03-21 10:15:00 | 962.90 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-03-20 10:15:00 | 913.70 | 2025-03-21 10:15:00 | 959.39 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-03-20 12:00:00 | 912.70 | 2025-03-21 10:15:00 | 958.34 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-03-20 09:15:00 | 917.05 | 2025-03-24 10:15:00 | 951.00 | STOP_HIT | 0.50 | 3.70% |
| BUY | retest1 | 2025-03-20 10:15:00 | 913.70 | 2025-03-24 10:15:00 | 951.00 | STOP_HIT | 0.50 | 4.08% |
| BUY | retest1 | 2025-03-20 12:00:00 | 912.70 | 2025-03-24 10:15:00 | 951.00 | STOP_HIT | 0.50 | 4.20% |
| SELL | retest2 | 2025-03-26 11:30:00 | 926.55 | 2025-03-27 13:15:00 | 951.30 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-03-27 12:00:00 | 928.55 | 2025-03-27 13:15:00 | 951.30 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-04-02 13:30:00 | 933.35 | 2025-04-03 10:15:00 | 951.40 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-04-03 09:45:00 | 931.70 | 2025-04-03 10:15:00 | 951.40 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-04-23 09:15:00 | 951.80 | 2025-04-23 09:15:00 | 930.65 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-04-24 15:00:00 | 969.80 | 2025-04-25 09:15:00 | 924.50 | STOP_HIT | 1.00 | -4.67% |
| SELL | retest2 | 2025-05-02 11:15:00 | 915.45 | 2025-05-05 09:15:00 | 936.75 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-05-23 13:30:00 | 987.45 | 2025-05-26 09:15:00 | 1002.25 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-05-30 13:30:00 | 1017.70 | 2025-06-03 10:15:00 | 1001.50 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-05-30 14:45:00 | 1018.00 | 2025-06-03 10:15:00 | 1001.50 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-06-02 09:30:00 | 1017.50 | 2025-06-03 10:15:00 | 1001.50 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-06-02 12:00:00 | 1017.50 | 2025-06-03 10:15:00 | 1001.50 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-06-11 15:15:00 | 1049.00 | 2025-06-12 10:15:00 | 1033.70 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-06-26 13:45:00 | 990.00 | 2025-07-03 12:15:00 | 1011.10 | STOP_HIT | 1.00 | 2.13% |
| BUY | retest2 | 2025-06-26 14:45:00 | 991.20 | 2025-07-03 12:15:00 | 1011.10 | STOP_HIT | 1.00 | 2.01% |
| SELL | retest2 | 2025-07-29 10:15:00 | 996.80 | 2025-07-29 14:15:00 | 1014.20 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-07-29 10:45:00 | 1002.00 | 2025-07-29 14:15:00 | 1014.20 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-08-05 09:15:00 | 987.15 | 2025-08-07 09:15:00 | 937.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 09:15:00 | 987.15 | 2025-08-07 14:15:00 | 938.40 | STOP_HIT | 0.50 | 4.94% |
| BUY | retest2 | 2025-10-07 09:15:00 | 1086.90 | 2025-10-08 09:15:00 | 1046.20 | STOP_HIT | 1.00 | -3.74% |
| BUY | retest2 | 2025-10-07 13:30:00 | 1063.00 | 2025-10-08 09:15:00 | 1046.20 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-10-07 14:00:00 | 1064.30 | 2025-10-08 09:15:00 | 1046.20 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest1 | 2025-11-03 09:15:00 | 1157.30 | 2025-11-03 13:15:00 | 1122.60 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-11-10 12:45:00 | 1064.10 | 2025-11-12 09:15:00 | 1088.30 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-11-10 13:30:00 | 1061.10 | 2025-11-12 09:15:00 | 1088.30 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-12-08 09:15:00 | 1013.60 | 2025-12-11 13:15:00 | 1013.40 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2025-12-10 10:00:00 | 1013.90 | 2025-12-11 13:15:00 | 1013.40 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2025-12-10 10:30:00 | 1014.20 | 2025-12-11 13:15:00 | 1013.40 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2025-12-17 13:30:00 | 1021.00 | 2025-12-22 11:15:00 | 1022.00 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-12-17 14:00:00 | 1020.40 | 2025-12-22 11:15:00 | 1022.00 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-12-19 15:15:00 | 1018.00 | 2025-12-22 11:15:00 | 1022.00 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-12-30 11:15:00 | 1001.50 | 2025-12-31 09:15:00 | 1013.60 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-12-30 13:15:00 | 1001.70 | 2025-12-31 09:15:00 | 1013.60 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2026-01-20 12:00:00 | 898.80 | 2026-01-23 12:15:00 | 853.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 12:30:00 | 895.90 | 2026-01-23 12:15:00 | 851.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 10:45:00 | 900.60 | 2026-01-23 12:15:00 | 855.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 11:30:00 | 897.30 | 2026-01-23 12:15:00 | 852.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 09:15:00 | 892.50 | 2026-01-23 12:15:00 | 847.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 12:00:00 | 898.80 | 2026-01-23 13:15:00 | 808.92 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-20 12:30:00 | 895.90 | 2026-01-23 13:15:00 | 806.31 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-22 10:45:00 | 900.60 | 2026-01-23 13:15:00 | 810.54 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-22 11:30:00 | 897.30 | 2026-01-23 13:15:00 | 807.57 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-23 09:15:00 | 892.50 | 2026-01-23 13:15:00 | 803.25 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-12 12:30:00 | 989.80 | 2026-02-13 09:15:00 | 950.80 | STOP_HIT | 1.00 | -3.94% |
| BUY | retest2 | 2026-02-12 13:15:00 | 990.55 | 2026-02-13 09:15:00 | 950.80 | STOP_HIT | 1.00 | -4.01% |
| BUY | retest2 | 2026-02-19 12:15:00 | 991.15 | 2026-02-19 13:15:00 | 984.50 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2026-02-19 12:45:00 | 990.30 | 2026-02-19 13:15:00 | 984.50 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2026-02-23 11:15:00 | 975.40 | 2026-03-02 09:15:00 | 926.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 09:30:00 | 974.65 | 2026-03-02 09:15:00 | 925.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 10:30:00 | 974.10 | 2026-03-02 09:15:00 | 925.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 11:15:00 | 975.40 | 2026-03-04 09:15:00 | 877.86 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-25 09:30:00 | 974.65 | 2026-03-04 09:15:00 | 877.18 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-25 10:30:00 | 974.10 | 2026-03-04 09:15:00 | 876.69 | TARGET_HIT | 0.50 | 10.00% |
