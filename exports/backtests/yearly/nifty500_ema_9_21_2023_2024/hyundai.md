# Hyundai Motor India Ltd. (HYUNDAI)

## Backtest Summary

- **Window:** 2024-10-22 09:15:00 → 2026-05-08 15:15:00 (2664 bars)
- **Last close:** 1833.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 129 |
| ALERT1 | 75 |
| ALERT2 | 74 |
| ALERT2_SKIP | 41 |
| ALERT3 | 167 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 85 |
| PARTIAL | 6 |
| TARGET_HIT | 1 |
| STOP_HIT | 83 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 90 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 16 / 74
- **Target hits / Stop hits / Partials:** 1 / 83 / 6
- **Avg / median % per leg:** -0.31% / -0.91%
- **Sum % (uncompounded):** -27.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 46 | 2 | 4.3% | 1 | 45 | 0 | -0.88% | -40.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 46 | 2 | 4.3% | 1 | 45 | 0 | -0.88% | -40.3% |
| SELL (all) | 44 | 14 | 31.8% | 0 | 38 | 6 | 0.28% | 12.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 44 | 14 | 31.8% | 0 | 38 | 6 | 0.28% | 12.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 90 | 16 | 17.8% | 1 | 83 | 6 | -0.31% | -27.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 14:15:00 | 1835.85 | 1808.26 | 1807.45 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 12:15:00 | 1809.10 | 1815.00 | 1815.25 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-04 13:15:00 | 1821.50 | 1816.30 | 1815.81 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 14:15:00 | 1810.90 | 1815.22 | 1815.37 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-04 15:15:00 | 1818.00 | 1815.78 | 1815.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 09:15:00 | 1821.75 | 1816.97 | 1816.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 1841.00 | 1846.43 | 1837.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 10:15:00 | 1837.55 | 1844.66 | 1837.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 1837.55 | 1844.66 | 1837.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:45:00 | 1837.70 | 1844.66 | 1837.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 1834.35 | 1842.59 | 1837.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:00:00 | 1834.35 | 1842.59 | 1837.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 1836.45 | 1841.37 | 1837.06 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 1821.25 | 1834.73 | 1835.34 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-11-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-08 14:15:00 | 1840.20 | 1835.11 | 1835.06 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 12:15:00 | 1832.70 | 1835.19 | 1835.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 13:15:00 | 1827.00 | 1833.55 | 1834.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 15:15:00 | 1820.00 | 1814.23 | 1822.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 15:15:00 | 1820.00 | 1814.23 | 1822.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 15:15:00 | 1820.00 | 1814.23 | 1822.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 09:15:00 | 1730.00 | 1814.23 | 1822.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 11:15:00 | 1831.05 | 1769.95 | 1762.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 1831.05 | 1769.95 | 1762.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 13:15:00 | 1836.60 | 1793.05 | 1774.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-22 13:15:00 | 1827.85 | 1827.87 | 1815.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-22 13:30:00 | 1827.75 | 1827.87 | 1815.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 1876.55 | 1894.81 | 1880.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:15:00 | 1890.10 | 1894.81 | 1880.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 1889.25 | 1893.70 | 1881.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 11:30:00 | 1901.20 | 1894.96 | 1883.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 13:15:00 | 1885.40 | 1894.70 | 1895.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 13:15:00 | 1885.40 | 1894.70 | 1895.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-03 09:15:00 | 1857.60 | 1883.39 | 1889.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 10:15:00 | 1861.20 | 1857.03 | 1868.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-04 11:00:00 | 1861.20 | 1857.03 | 1868.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 1870.70 | 1860.65 | 1868.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:30:00 | 1869.05 | 1860.65 | 1868.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 13:15:00 | 1871.45 | 1862.81 | 1868.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 14:00:00 | 1871.45 | 1862.81 | 1868.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 1871.00 | 1864.45 | 1868.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 15:00:00 | 1871.00 | 1864.45 | 1868.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 15:15:00 | 1873.00 | 1866.16 | 1869.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:15:00 | 1885.00 | 1866.16 | 1869.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2024-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 11:15:00 | 1881.00 | 1871.57 | 1871.14 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-12-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 10:15:00 | 1857.00 | 1869.13 | 1870.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 09:15:00 | 1843.25 | 1857.79 | 1863.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 14:15:00 | 1833.80 | 1832.15 | 1840.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-10 14:45:00 | 1833.60 | 1832.15 | 1840.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 1825.00 | 1830.63 | 1838.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 09:15:00 | 1798.50 | 1825.60 | 1832.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 14:15:00 | 1814.85 | 1796.47 | 1796.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 14:15:00 | 1814.85 | 1796.47 | 1796.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 11:15:00 | 1824.55 | 1810.39 | 1803.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 09:15:00 | 1798.85 | 1812.73 | 1808.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 09:15:00 | 1798.85 | 1812.73 | 1808.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 1798.85 | 1812.73 | 1808.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 10:00:00 | 1798.85 | 1812.73 | 1808.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 1800.90 | 1810.36 | 1807.36 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2024-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 13:15:00 | 1795.80 | 1803.68 | 1804.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 1779.90 | 1796.45 | 1800.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 14:15:00 | 1786.95 | 1786.40 | 1793.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 15:00:00 | 1786.95 | 1786.40 | 1793.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 1767.50 | 1782.40 | 1790.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 10:15:00 | 1759.50 | 1782.40 | 1790.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 09:15:00 | 1748.20 | 1773.65 | 1781.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 15:00:00 | 1764.30 | 1759.68 | 1764.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 09:15:00 | 1759.95 | 1760.74 | 1764.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 1754.35 | 1759.46 | 1763.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 10:15:00 | 1751.05 | 1759.46 | 1763.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-26 12:15:00 | 1779.45 | 1765.83 | 1765.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 12:15:00 | 1779.45 | 1765.83 | 1765.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 1794.05 | 1775.98 | 1771.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 10:15:00 | 1793.60 | 1797.60 | 1790.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-31 11:00:00 | 1793.60 | 1797.60 | 1790.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 1800.00 | 1800.71 | 1795.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:45:00 | 1794.90 | 1800.71 | 1795.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 11:15:00 | 1796.95 | 1799.86 | 1795.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 12:00:00 | 1796.95 | 1799.86 | 1795.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 12:15:00 | 1800.00 | 1799.89 | 1796.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 12:30:00 | 1799.00 | 1799.89 | 1796.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 15:15:00 | 1798.00 | 1799.59 | 1797.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 09:15:00 | 1801.85 | 1799.59 | 1797.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 1813.90 | 1802.45 | 1798.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 12:15:00 | 1841.45 | 1802.58 | 1799.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 13:30:00 | 1825.00 | 1809.71 | 1803.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 09:15:00 | 1821.20 | 1812.49 | 1805.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 09:45:00 | 1821.75 | 1815.19 | 1807.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 1821.30 | 1823.26 | 1816.25 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-06 14:15:00 | 1800.15 | 1812.13 | 1813.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 14:15:00 | 1800.15 | 1812.13 | 1813.09 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 10:15:00 | 1835.80 | 1814.97 | 1813.92 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 12:15:00 | 1804.65 | 1815.57 | 1816.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 09:15:00 | 1774.50 | 1804.90 | 1811.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 15:15:00 | 1788.40 | 1788.16 | 1798.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-10 09:15:00 | 1791.75 | 1788.16 | 1798.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 1788.95 | 1788.32 | 1797.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 09:15:00 | 1762.15 | 1788.41 | 1793.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 12:00:00 | 1771.05 | 1779.46 | 1787.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-14 10:15:00 | 1803.00 | 1783.04 | 1784.94 | SL hit (close>static) qty=1.00 sl=1798.80 alert=retest2 |

### Cycle 19 — BUY (started 2025-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 12:15:00 | 1797.00 | 1787.60 | 1786.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-14 13:15:00 | 1798.30 | 1789.74 | 1787.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-14 14:15:00 | 1780.05 | 1787.80 | 1787.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 14:15:00 | 1780.05 | 1787.80 | 1787.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 1780.05 | 1787.80 | 1787.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-14 15:00:00 | 1780.05 | 1787.80 | 1787.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 1790.00 | 1788.24 | 1787.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:30:00 | 1783.10 | 1787.69 | 1787.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 10:15:00 | 1778.60 | 1785.88 | 1786.44 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 11:15:00 | 1800.05 | 1788.71 | 1787.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 1812.00 | 1798.21 | 1793.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 1799.40 | 1808.82 | 1802.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 09:15:00 | 1799.40 | 1808.82 | 1802.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 1799.40 | 1808.82 | 1802.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 10:30:00 | 1807.15 | 1807.07 | 1802.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 11:15:00 | 1802.15 | 1807.07 | 1802.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 12:45:00 | 1803.90 | 1805.95 | 1802.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-17 15:15:00 | 1790.00 | 1800.93 | 1801.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-01-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 15:15:00 | 1790.00 | 1800.93 | 1801.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 10:15:00 | 1769.55 | 1783.84 | 1790.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-24 11:15:00 | 1706.10 | 1703.01 | 1716.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-24 12:00:00 | 1706.10 | 1703.01 | 1716.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 1648.90 | 1639.78 | 1654.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 13:00:00 | 1648.90 | 1639.78 | 1654.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 13:15:00 | 1648.90 | 1641.60 | 1654.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 13:30:00 | 1648.80 | 1641.60 | 1654.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 1646.65 | 1642.61 | 1653.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 15:00:00 | 1646.65 | 1642.61 | 1653.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 15:15:00 | 1645.00 | 1643.09 | 1652.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:15:00 | 1646.40 | 1643.09 | 1652.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 1647.25 | 1643.92 | 1652.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 10:15:00 | 1644.60 | 1643.92 | 1652.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-31 10:15:00 | 1661.70 | 1639.90 | 1643.71 | SL hit (close>static) qty=1.00 sl=1655.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-01-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 12:15:00 | 1676.00 | 1651.41 | 1648.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 13:15:00 | 1685.00 | 1658.13 | 1651.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 09:15:00 | 1815.65 | 1824.38 | 1791.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-05 09:30:00 | 1811.05 | 1824.38 | 1791.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 1801.00 | 1824.89 | 1808.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:45:00 | 1794.75 | 1824.89 | 1808.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 1812.05 | 1822.32 | 1808.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:45:00 | 1812.15 | 1822.32 | 1808.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 1820.00 | 1823.43 | 1812.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:00:00 | 1820.00 | 1823.43 | 1812.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1885.50 | 1836.33 | 1821.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:45:00 | 1890.60 | 1849.76 | 1828.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-11 13:15:00 | 1816.85 | 1846.45 | 1850.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-02-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 13:15:00 | 1816.85 | 1846.45 | 1850.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-13 13:15:00 | 1796.00 | 1816.34 | 1824.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-14 09:15:00 | 1840.00 | 1814.85 | 1821.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-14 09:15:00 | 1840.00 | 1814.85 | 1821.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 1840.00 | 1814.85 | 1821.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-14 10:00:00 | 1840.00 | 1814.85 | 1821.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 10:15:00 | 1820.00 | 1815.88 | 1821.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 11:30:00 | 1802.00 | 1812.54 | 1819.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 1851.55 | 1815.59 | 1813.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 14:15:00 | 1851.55 | 1815.59 | 1813.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 09:15:00 | 1856.75 | 1827.41 | 1819.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-18 10:15:00 | 1820.15 | 1825.96 | 1819.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-18 11:00:00 | 1820.15 | 1825.96 | 1819.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 11:15:00 | 1808.60 | 1822.48 | 1818.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-18 12:00:00 | 1808.60 | 1822.48 | 1818.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 12:15:00 | 1805.70 | 1819.13 | 1817.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-18 12:45:00 | 1802.65 | 1819.13 | 1817.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 14:15:00 | 1810.95 | 1815.43 | 1815.82 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 10:15:00 | 1825.50 | 1816.73 | 1816.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 11:15:00 | 1862.55 | 1825.89 | 1820.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 10:15:00 | 1842.90 | 1850.95 | 1837.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-20 10:15:00 | 1842.90 | 1850.95 | 1837.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 10:15:00 | 1842.90 | 1850.95 | 1837.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 10:45:00 | 1839.10 | 1850.95 | 1837.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 13:15:00 | 1841.50 | 1848.32 | 1839.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 13:30:00 | 1838.40 | 1848.32 | 1839.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 1827.65 | 1845.62 | 1840.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:15:00 | 1823.70 | 1845.62 | 1840.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 1819.10 | 1840.31 | 1838.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:45:00 | 1820.00 | 1840.31 | 1838.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 11:15:00 | 1796.70 | 1831.59 | 1835.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 15:15:00 | 1789.00 | 1813.48 | 1824.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 1832.95 | 1804.58 | 1811.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 09:15:00 | 1832.95 | 1804.58 | 1811.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 1832.95 | 1804.58 | 1811.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 10:00:00 | 1832.95 | 1804.58 | 1811.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 1817.95 | 1807.25 | 1812.45 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2025-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 14:15:00 | 1823.30 | 1815.03 | 1814.87 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 1743.05 | 1802.85 | 1809.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 10:15:00 | 1708.15 | 1783.91 | 1800.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 09:15:00 | 1752.85 | 1739.32 | 1765.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 09:15:00 | 1752.85 | 1739.32 | 1765.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 1752.85 | 1739.32 | 1765.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 12:15:00 | 1726.50 | 1740.13 | 1761.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-06 10:15:00 | 1720.60 | 1715.91 | 1715.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 10:15:00 | 1720.60 | 1715.91 | 1715.80 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-06 11:15:00 | 1693.00 | 1711.33 | 1713.73 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-03-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-07 09:15:00 | 1718.00 | 1715.03 | 1714.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 10:15:00 | 1720.90 | 1716.20 | 1715.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 1710.20 | 1715.00 | 1714.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 11:15:00 | 1710.20 | 1715.00 | 1714.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 1710.20 | 1715.00 | 1714.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:00:00 | 1710.20 | 1715.00 | 1714.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-03-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 12:15:00 | 1711.00 | 1714.20 | 1714.59 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-03-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-07 15:15:00 | 1718.00 | 1714.96 | 1714.85 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 09:15:00 | 1689.05 | 1709.78 | 1712.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 10:15:00 | 1680.50 | 1703.92 | 1709.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 1667.45 | 1657.79 | 1667.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 09:15:00 | 1667.45 | 1657.79 | 1667.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 1667.45 | 1657.79 | 1667.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:30:00 | 1670.00 | 1657.79 | 1667.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 1664.45 | 1659.12 | 1667.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 12:00:00 | 1659.75 | 1659.25 | 1666.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 15:00:00 | 1639.25 | 1653.84 | 1662.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 09:15:00 | 1576.76 | 1604.99 | 1627.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 10:15:00 | 1557.29 | 1596.24 | 1621.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-18 15:15:00 | 1582.00 | 1579.41 | 1601.48 | SL hit (close>ema200) qty=0.50 sl=1579.41 alert=retest2 |

### Cycle 37 — BUY (started 2025-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 09:15:00 | 1616.40 | 1609.56 | 1608.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-21 11:15:00 | 1671.00 | 1640.41 | 1627.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 10:15:00 | 1721.65 | 1728.84 | 1699.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 11:00:00 | 1721.65 | 1728.84 | 1699.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 1693.70 | 1724.20 | 1713.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 12:00:00 | 1693.70 | 1724.20 | 1713.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 12:15:00 | 1691.35 | 1717.63 | 1711.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 14:00:00 | 1700.50 | 1714.20 | 1710.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 09:45:00 | 1709.85 | 1712.44 | 1710.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 11:15:00 | 1700.00 | 1708.84 | 1709.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 11:15:00 | 1700.00 | 1708.84 | 1709.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 12:15:00 | 1690.55 | 1705.18 | 1707.39 | Break + close below crossover candle low |

### Cycle 39 — BUY (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 14:15:00 | 1746.25 | 1712.32 | 1710.18 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 12:15:00 | 1709.45 | 1712.83 | 1713.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 14:15:00 | 1703.70 | 1710.42 | 1711.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-04 09:15:00 | 1658.70 | 1657.76 | 1672.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-04 09:30:00 | 1658.60 | 1657.76 | 1672.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 1623.85 | 1599.65 | 1602.80 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 1632.05 | 1606.13 | 1605.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1646.10 | 1624.38 | 1615.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 09:15:00 | 1637.60 | 1644.03 | 1632.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 09:45:00 | 1638.30 | 1644.03 | 1632.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 10:15:00 | 1650.00 | 1645.23 | 1634.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 10:30:00 | 1637.60 | 1645.23 | 1634.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 13:15:00 | 1637.20 | 1641.25 | 1634.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 13:30:00 | 1633.70 | 1641.25 | 1634.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 14:15:00 | 1636.00 | 1640.20 | 1634.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 14:45:00 | 1633.50 | 1640.20 | 1634.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 15:15:00 | 1639.50 | 1640.06 | 1635.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 09:15:00 | 1648.00 | 1640.06 | 1635.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 10:15:00 | 1668.70 | 1699.82 | 1699.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 1668.70 | 1699.82 | 1699.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 09:15:00 | 1657.90 | 1674.16 | 1684.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 15:15:00 | 1663.20 | 1662.93 | 1673.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-29 09:15:00 | 1672.20 | 1662.93 | 1673.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 1670.80 | 1664.50 | 1673.06 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 09:15:00 | 1699.30 | 1679.53 | 1677.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 14:15:00 | 1724.70 | 1707.94 | 1698.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 1730.40 | 1737.06 | 1727.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 09:15:00 | 1730.40 | 1737.06 | 1727.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 1730.40 | 1737.06 | 1727.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 10:15:00 | 1736.10 | 1737.06 | 1727.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 1699.70 | 1740.82 | 1743.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 1699.70 | 1740.82 | 1743.91 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1753.70 | 1741.29 | 1740.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 1775.20 | 1755.47 | 1748.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 14:15:00 | 1764.70 | 1764.95 | 1756.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 15:00:00 | 1764.70 | 1764.95 | 1756.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 1885.10 | 1897.80 | 1883.44 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 14:15:00 | 1851.80 | 1874.10 | 1876.14 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 1905.60 | 1879.31 | 1875.74 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 10:15:00 | 1864.40 | 1879.75 | 1880.17 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 15:15:00 | 1887.10 | 1881.04 | 1880.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 10:15:00 | 1888.00 | 1883.46 | 1881.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 12:15:00 | 1883.00 | 1884.42 | 1882.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 12:15:00 | 1883.00 | 1884.42 | 1882.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 1883.00 | 1884.42 | 1882.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 13:00:00 | 1883.00 | 1884.42 | 1882.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 1892.70 | 1886.07 | 1883.37 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 13:15:00 | 1866.10 | 1881.65 | 1882.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 1852.40 | 1870.14 | 1875.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 14:15:00 | 1860.90 | 1858.01 | 1864.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 14:15:00 | 1860.90 | 1858.01 | 1864.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 1860.90 | 1858.01 | 1864.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 1860.90 | 1858.01 | 1864.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 1852.80 | 1856.96 | 1863.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 09:15:00 | 1844.10 | 1856.96 | 1863.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 12:15:00 | 1855.10 | 1837.42 | 1836.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 12:15:00 | 1855.10 | 1837.42 | 1836.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 14:15:00 | 1863.00 | 1846.13 | 1840.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 12:15:00 | 1935.60 | 1939.27 | 1922.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 13:00:00 | 1935.60 | 1939.27 | 1922.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 1943.10 | 1950.07 | 1940.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 1944.80 | 1950.07 | 1940.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 1959.30 | 1941.63 | 1939.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 10:15:00 | 1947.50 | 1941.50 | 1939.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 11:30:00 | 1947.70 | 1942.27 | 1940.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 1936.30 | 1941.08 | 1940.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 13:45:00 | 1939.30 | 1940.72 | 1940.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 14:30:00 | 1940.80 | 1940.56 | 1940.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 1962.40 | 1939.85 | 1939.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 10:15:00 | 1931.10 | 1940.20 | 1941.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 10:15:00 | 1931.10 | 1940.20 | 1941.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 11:15:00 | 1929.50 | 1938.06 | 1940.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 1938.70 | 1917.66 | 1923.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 1938.70 | 1917.66 | 1923.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1938.70 | 1917.66 | 1923.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:30:00 | 1947.30 | 1917.66 | 1923.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1958.90 | 1925.90 | 1926.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:45:00 | 1955.60 | 1925.90 | 1926.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 11:15:00 | 1976.10 | 1935.94 | 1931.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 14:15:00 | 2022.10 | 1963.25 | 1945.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 10:15:00 | 2101.00 | 2107.98 | 2074.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 11:00:00 | 2101.00 | 2107.98 | 2074.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 2141.10 | 2114.51 | 2095.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 15:00:00 | 2187.90 | 2134.78 | 2109.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 13:00:00 | 2161.10 | 2204.23 | 2198.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 13:15:00 | 2141.70 | 2191.73 | 2193.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 13:15:00 | 2141.70 | 2191.73 | 2193.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 14:15:00 | 2136.00 | 2180.58 | 2187.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 09:15:00 | 2075.00 | 2073.60 | 2099.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 11:15:00 | 2089.70 | 2079.88 | 2097.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 2089.70 | 2079.88 | 2097.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:30:00 | 2102.00 | 2079.88 | 2097.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 2100.00 | 2085.01 | 2095.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 15:00:00 | 2100.00 | 2085.01 | 2095.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 2098.00 | 2087.61 | 2096.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:15:00 | 2080.30 | 2087.61 | 2096.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 2067.70 | 2083.63 | 2093.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 10:30:00 | 2060.50 | 2078.22 | 2090.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 09:30:00 | 2063.90 | 2061.20 | 2074.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 10:15:00 | 2090.30 | 2078.83 | 2077.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 10:15:00 | 2090.30 | 2078.83 | 2077.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 11:15:00 | 2098.80 | 2082.82 | 2079.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 10:15:00 | 2094.10 | 2104.22 | 2097.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 10:15:00 | 2094.10 | 2104.22 | 2097.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 2094.10 | 2104.22 | 2097.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:00:00 | 2094.10 | 2104.22 | 2097.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 2123.00 | 2107.97 | 2100.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 12:15:00 | 2136.60 | 2107.97 | 2100.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 12:00:00 | 2140.50 | 2135.88 | 2121.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 14:00:00 | 2133.40 | 2130.20 | 2127.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 10:15:00 | 2127.00 | 2129.84 | 2128.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 2131.20 | 2130.11 | 2128.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 12:00:00 | 2138.20 | 2131.73 | 2129.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 10:15:00 | 2125.00 | 2129.50 | 2129.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 2125.00 | 2129.50 | 2129.54 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 11:15:00 | 2135.00 | 2130.60 | 2130.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 12:15:00 | 2142.50 | 2132.98 | 2131.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 13:15:00 | 2128.60 | 2132.11 | 2130.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 13:15:00 | 2128.60 | 2132.11 | 2130.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 2128.60 | 2132.11 | 2130.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:15:00 | 2126.00 | 2132.11 | 2130.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 14:15:00 | 2121.90 | 2130.06 | 2130.11 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 15:15:00 | 2140.00 | 2127.71 | 2127.33 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 09:15:00 | 2108.40 | 2123.85 | 2125.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 12:15:00 | 2100.30 | 2112.28 | 2116.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 14:15:00 | 2103.20 | 2094.91 | 2101.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 14:15:00 | 2103.20 | 2094.91 | 2101.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 2103.20 | 2094.91 | 2101.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 15:00:00 | 2103.20 | 2094.91 | 2101.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 2090.00 | 2093.93 | 2100.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 09:15:00 | 2076.50 | 2093.93 | 2100.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 09:15:00 | 2080.00 | 2082.17 | 2083.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 14:15:00 | 2092.80 | 2083.14 | 2083.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 14:15:00 | 2092.80 | 2083.14 | 2083.13 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 15:15:00 | 2082.00 | 2082.91 | 2083.02 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 09:15:00 | 2091.60 | 2084.65 | 2083.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 10:15:00 | 2110.00 | 2089.72 | 2086.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-04 09:15:00 | 2151.40 | 2163.58 | 2142.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-04 09:45:00 | 2158.00 | 2163.58 | 2142.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 2192.80 | 2184.50 | 2170.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 14:45:00 | 2170.70 | 2184.50 | 2170.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 2142.30 | 2176.46 | 2169.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 2142.30 | 2176.46 | 2169.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 2135.50 | 2168.27 | 2166.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:45:00 | 2134.80 | 2168.27 | 2166.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 2145.50 | 2163.71 | 2164.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 2122.70 | 2144.59 | 2153.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 10:15:00 | 2134.80 | 2126.22 | 2133.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 10:15:00 | 2134.80 | 2126.22 | 2133.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 2134.80 | 2126.22 | 2133.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:45:00 | 2140.90 | 2126.22 | 2133.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 2141.20 | 2129.21 | 2134.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:00:00 | 2141.20 | 2129.21 | 2134.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 2136.80 | 2130.73 | 2134.46 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 2151.50 | 2138.74 | 2137.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 2220.40 | 2157.05 | 2146.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 12:15:00 | 2224.00 | 2232.39 | 2215.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 13:00:00 | 2224.00 | 2232.39 | 2215.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 2503.50 | 2504.09 | 2467.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 2456.30 | 2504.09 | 2467.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 2466.70 | 2493.15 | 2471.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:00:00 | 2466.70 | 2493.15 | 2471.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 2469.80 | 2488.48 | 2471.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:45:00 | 2470.80 | 2488.48 | 2471.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 2449.50 | 2480.69 | 2469.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 2449.50 | 2480.69 | 2469.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 2445.00 | 2473.55 | 2467.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 2435.50 | 2473.55 | 2467.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 2431.20 | 2460.54 | 2462.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 12:15:00 | 2409.20 | 2445.56 | 2454.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 2414.60 | 2409.90 | 2432.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 10:00:00 | 2414.60 | 2409.90 | 2432.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 2446.80 | 2417.55 | 2431.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:00:00 | 2446.80 | 2417.55 | 2431.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 2474.50 | 2428.94 | 2435.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:00:00 | 2474.50 | 2428.94 | 2435.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2025-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 14:15:00 | 2466.40 | 2442.31 | 2440.92 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 2425.50 | 2440.89 | 2441.04 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 12:15:00 | 2468.90 | 2446.29 | 2443.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-26 14:15:00 | 2495.90 | 2459.12 | 2449.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-28 12:15:00 | 2468.40 | 2476.31 | 2463.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 12:15:00 | 2468.40 | 2476.31 | 2463.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 12:15:00 | 2468.40 | 2476.31 | 2463.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 12:45:00 | 2477.30 | 2476.31 | 2463.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 13:15:00 | 2465.00 | 2474.05 | 2463.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 13:30:00 | 2465.60 | 2474.05 | 2463.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 2452.00 | 2469.64 | 2462.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 15:00:00 | 2452.00 | 2469.64 | 2462.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 2440.00 | 2463.71 | 2460.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 09:30:00 | 2459.60 | 2460.35 | 2459.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 10:15:00 | 2465.00 | 2460.35 | 2459.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 12:15:00 | 2454.00 | 2458.29 | 2458.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 12:15:00 | 2454.00 | 2458.29 | 2458.71 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 2468.10 | 2459.76 | 2459.19 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 10:15:00 | 2450.00 | 2457.81 | 2458.35 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 12:15:00 | 2474.80 | 2461.37 | 2459.88 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 09:15:00 | 2439.10 | 2455.76 | 2457.88 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 2498.90 | 2460.76 | 2458.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 12:15:00 | 2529.70 | 2505.53 | 2493.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 15:15:00 | 2552.80 | 2557.45 | 2536.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 09:15:00 | 2575.00 | 2557.45 | 2536.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 2595.90 | 2565.14 | 2541.48 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 09:15:00 | 2489.00 | 2534.39 | 2536.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 11:15:00 | 2474.70 | 2515.19 | 2527.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 13:15:00 | 2510.40 | 2501.52 | 2510.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 13:15:00 | 2510.40 | 2501.52 | 2510.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 2510.40 | 2501.52 | 2510.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:00:00 | 2510.40 | 2501.52 | 2510.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 2510.00 | 2503.22 | 2510.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 2510.00 | 2503.22 | 2510.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 2498.10 | 2502.20 | 2509.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 2544.10 | 2502.20 | 2509.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 2536.60 | 2509.08 | 2511.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:45:00 | 2540.10 | 2509.08 | 2511.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 10:15:00 | 2544.60 | 2516.18 | 2514.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 14:15:00 | 2561.40 | 2535.83 | 2525.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 09:15:00 | 2538.90 | 2539.99 | 2529.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 10:00:00 | 2538.90 | 2539.99 | 2529.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 2527.70 | 2536.74 | 2529.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:00:00 | 2527.70 | 2536.74 | 2529.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 2536.10 | 2536.61 | 2530.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 13:15:00 | 2542.50 | 2536.61 | 2530.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-19 14:15:00 | 2796.75 | 2744.08 | 2693.59 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 09:15:00 | 2723.80 | 2730.48 | 2730.78 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 13:15:00 | 2761.80 | 2733.99 | 2731.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-24 15:15:00 | 2767.00 | 2742.04 | 2735.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 09:15:00 | 2730.10 | 2739.65 | 2735.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 2730.10 | 2739.65 | 2735.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 2730.10 | 2739.65 | 2735.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 11:15:00 | 2764.50 | 2740.32 | 2735.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 15:15:00 | 2715.00 | 2736.89 | 2736.63 | SL hit (close<static) qty=1.00 sl=2715.80 alert=retest2 |

### Cycle 80 — SELL (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 09:15:00 | 2671.20 | 2723.75 | 2730.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 14:15:00 | 2643.30 | 2683.39 | 2706.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 2693.00 | 2677.57 | 2699.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 2693.00 | 2677.57 | 2699.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 2670.00 | 2676.06 | 2696.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 2648.80 | 2668.84 | 2691.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 11:15:00 | 2516.36 | 2576.95 | 2616.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 10:15:00 | 2556.70 | 2549.20 | 2582.32 | SL hit (close>ema200) qty=0.50 sl=2549.20 alert=retest2 |

### Cycle 81 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 2469.20 | 2413.41 | 2409.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 11:15:00 | 2471.90 | 2425.11 | 2415.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 14:15:00 | 2420.00 | 2429.60 | 2420.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 14:15:00 | 2420.00 | 2429.60 | 2420.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 2420.00 | 2429.60 | 2420.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 2420.00 | 2429.60 | 2420.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 2419.40 | 2427.56 | 2420.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 2441.80 | 2427.56 | 2420.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 10:15:00 | 2376.50 | 2417.64 | 2417.23 | SL hit (close<static) qty=1.00 sl=2415.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 11:15:00 | 2377.30 | 2409.58 | 2413.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-16 13:15:00 | 2358.80 | 2394.11 | 2405.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 14:15:00 | 2348.70 | 2342.19 | 2366.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-17 15:00:00 | 2348.70 | 2342.19 | 2366.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 2354.30 | 2345.70 | 2363.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:30:00 | 2359.80 | 2345.70 | 2363.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 2320.40 | 2333.90 | 2345.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 14:00:00 | 2299.00 | 2320.47 | 2335.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 13:15:00 | 2306.50 | 2289.24 | 2288.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 13:15:00 | 2306.50 | 2289.24 | 2288.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 09:15:00 | 2336.20 | 2301.21 | 2294.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 10:15:00 | 2342.10 | 2342.48 | 2325.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 10:30:00 | 2346.10 | 2342.48 | 2325.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 2403.00 | 2419.88 | 2404.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:30:00 | 2404.60 | 2419.88 | 2404.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 2412.80 | 2418.46 | 2405.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:45:00 | 2400.40 | 2418.46 | 2405.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 2377.50 | 2409.80 | 2404.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:00:00 | 2377.50 | 2409.80 | 2404.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 2389.50 | 2405.74 | 2403.08 | EMA400 retest candle locked (from upside) |

### Cycle 84 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 2367.20 | 2396.14 | 2399.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 14:15:00 | 2353.60 | 2379.03 | 2389.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 10:15:00 | 2362.20 | 2341.66 | 2356.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 10:15:00 | 2362.20 | 2341.66 | 2356.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 2362.20 | 2341.66 | 2356.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:00:00 | 2362.20 | 2341.66 | 2356.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 2360.00 | 2345.33 | 2356.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 15:15:00 | 2350.00 | 2355.96 | 2359.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 15:00:00 | 2355.10 | 2350.35 | 2353.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 2382.60 | 2358.14 | 2356.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 2382.60 | 2358.14 | 2356.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 10:15:00 | 2390.70 | 2364.65 | 2359.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 2411.00 | 2416.24 | 2397.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 15:00:00 | 2411.00 | 2416.24 | 2397.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 2417.00 | 2414.37 | 2401.25 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2025-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 14:15:00 | 2357.90 | 2392.38 | 2394.50 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 12:15:00 | 2397.70 | 2388.61 | 2387.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 09:15:00 | 2414.60 | 2395.87 | 2391.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 09:15:00 | 2377.00 | 2405.92 | 2400.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 2377.00 | 2405.92 | 2400.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 2377.00 | 2405.92 | 2400.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 2377.00 | 2405.92 | 2400.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 10:15:00 | 2360.50 | 2396.84 | 2397.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 12:15:00 | 2354.00 | 2383.59 | 2390.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 14:15:00 | 2331.00 | 2323.42 | 2347.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 14:15:00 | 2331.00 | 2323.42 | 2347.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 2331.00 | 2323.42 | 2347.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 15:00:00 | 2331.00 | 2323.42 | 2347.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 2332.00 | 2325.14 | 2345.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 2325.90 | 2325.14 | 2345.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 15:00:00 | 2291.00 | 2314.89 | 2331.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 10:45:00 | 2329.80 | 2318.13 | 2328.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 10:15:00 | 2328.00 | 2311.56 | 2309.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 10:15:00 | 2328.00 | 2311.56 | 2309.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 13:15:00 | 2335.20 | 2320.90 | 2314.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 09:15:00 | 2351.00 | 2372.86 | 2352.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 09:15:00 | 2351.00 | 2372.86 | 2352.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 2351.00 | 2372.86 | 2352.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 2351.00 | 2372.86 | 2352.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 2359.10 | 2370.11 | 2353.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:30:00 | 2364.40 | 2370.11 | 2353.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 2355.00 | 2367.09 | 2353.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:00:00 | 2355.00 | 2367.09 | 2353.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 2371.40 | 2367.95 | 2355.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 13:15:00 | 2378.50 | 2367.95 | 2355.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 2341.00 | 2368.78 | 2360.53 | SL hit (close<static) qty=1.00 sl=2353.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 09:15:00 | 2347.00 | 2370.48 | 2370.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 10:15:00 | 2336.80 | 2363.75 | 2367.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 2286.30 | 2282.40 | 2303.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 2286.30 | 2282.40 | 2303.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 2285.80 | 2281.95 | 2295.18 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 15:15:00 | 2313.00 | 2298.13 | 2297.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 10:15:00 | 2322.90 | 2306.26 | 2301.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 09:15:00 | 2320.10 | 2323.79 | 2313.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 2320.10 | 2323.79 | 2313.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 2320.10 | 2323.79 | 2313.98 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 11:15:00 | 2295.00 | 2311.95 | 2313.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 2282.00 | 2299.25 | 2306.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 09:15:00 | 2280.00 | 2276.45 | 2288.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 2280.00 | 2276.45 | 2288.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 2280.00 | 2276.45 | 2288.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 2266.80 | 2280.82 | 2286.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 10:45:00 | 2269.70 | 2275.60 | 2282.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 11:15:00 | 2309.10 | 2287.74 | 2285.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 2309.10 | 2287.74 | 2285.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 12:15:00 | 2314.30 | 2293.05 | 2287.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 2313.00 | 2313.67 | 2305.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 09:15:00 | 2313.00 | 2313.67 | 2305.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 2313.00 | 2313.67 | 2305.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 2305.60 | 2313.67 | 2305.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 2313.00 | 2312.62 | 2307.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:30:00 | 2310.10 | 2312.62 | 2307.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 2310.40 | 2312.22 | 2308.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 15:00:00 | 2324.30 | 2316.71 | 2313.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 2302.60 | 2314.41 | 2312.67 | SL hit (close<static) qty=1.00 sl=2305.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 10:15:00 | 2299.50 | 2311.43 | 2311.48 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 11:15:00 | 2316.10 | 2312.36 | 2311.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 14:15:00 | 2323.00 | 2316.27 | 2313.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 15:15:00 | 2314.10 | 2315.84 | 2313.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 15:15:00 | 2314.10 | 2315.84 | 2313.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 2314.10 | 2315.84 | 2313.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 2297.80 | 2315.84 | 2313.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 2303.50 | 2313.37 | 2313.01 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 10:15:00 | 2307.00 | 2312.10 | 2312.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-31 09:15:00 | 2295.60 | 2303.55 | 2307.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 14:15:00 | 2299.00 | 2297.72 | 2302.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 15:00:00 | 2299.00 | 2297.72 | 2302.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 2293.10 | 2296.75 | 2301.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 10:15:00 | 2286.80 | 2296.75 | 2301.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 11:30:00 | 2290.00 | 2294.64 | 2299.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:00:00 | 2290.00 | 2294.64 | 2299.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 2267.90 | 2300.35 | 2300.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 2281.00 | 2276.04 | 2281.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:00:00 | 2281.00 | 2276.04 | 2281.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 2279.00 | 2276.63 | 2281.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 2280.60 | 2276.63 | 2281.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 2284.00 | 2278.10 | 2281.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:45:00 | 2289.80 | 2278.10 | 2281.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 2293.50 | 2281.18 | 2282.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:00:00 | 2293.50 | 2281.18 | 2282.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-06 11:15:00 | 2295.90 | 2284.13 | 2283.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 11:15:00 | 2295.90 | 2284.13 | 2283.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 14:15:00 | 2309.00 | 2291.89 | 2287.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 2330.00 | 2339.74 | 2322.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 11:00:00 | 2330.00 | 2339.74 | 2322.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 2267.50 | 2327.75 | 2324.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:00:00 | 2267.50 | 2327.75 | 2324.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2026-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 10:15:00 | 2262.90 | 2314.78 | 2318.95 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2026-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 15:15:00 | 2302.00 | 2288.26 | 2288.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 12:15:00 | 2315.90 | 2297.34 | 2292.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 2329.80 | 2330.14 | 2317.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 09:15:00 | 2329.80 | 2330.14 | 2317.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 2329.80 | 2330.14 | 2317.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 10:45:00 | 2345.50 | 2333.13 | 2319.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 12:15:00 | 2350.10 | 2333.70 | 2321.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 13:30:00 | 2345.70 | 2336.77 | 2324.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 14:30:00 | 2346.20 | 2338.48 | 2326.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 2309.80 | 2334.11 | 2326.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:30:00 | 2291.60 | 2334.11 | 2326.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 2306.60 | 2328.60 | 2325.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:30:00 | 2305.70 | 2328.60 | 2325.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 2324.80 | 2327.09 | 2324.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 13:00:00 | 2324.80 | 2327.09 | 2324.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 13:15:00 | 2316.80 | 2325.03 | 2324.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 14:00:00 | 2316.80 | 2325.03 | 2324.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-20 14:15:00 | 2316.00 | 2323.23 | 2323.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2026-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 14:15:00 | 2316.00 | 2323.23 | 2323.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 09:15:00 | 2258.40 | 2309.26 | 2317.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 13:15:00 | 2276.00 | 2274.06 | 2286.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 13:30:00 | 2275.50 | 2274.06 | 2286.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 2250.00 | 2269.65 | 2281.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:30:00 | 2246.50 | 2262.97 | 2276.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 2185.00 | 2261.81 | 2271.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-28 09:15:00 | 2134.17 | 2188.63 | 2221.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-29 14:15:00 | 2150.20 | 2145.91 | 2168.13 | SL hit (close>ema200) qty=0.50 sl=2145.91 alert=retest2 |

### Cycle 101 — BUY (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 09:15:00 | 2209.80 | 2173.58 | 2169.91 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 2140.40 | 2166.81 | 2169.79 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 2200.90 | 2175.24 | 2172.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 15:15:00 | 2210.00 | 2182.19 | 2176.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 2174.00 | 2193.95 | 2188.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 10:15:00 | 2174.00 | 2193.95 | 2188.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 2174.00 | 2193.95 | 2188.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:00:00 | 2174.00 | 2193.95 | 2188.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 2185.00 | 2192.16 | 2188.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 13:45:00 | 2190.60 | 2190.66 | 2188.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 15:00:00 | 2190.60 | 2190.65 | 2188.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 09:15:00 | 2151.20 | 2182.50 | 2184.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 2151.20 | 2182.50 | 2184.96 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2026-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 15:15:00 | 2185.00 | 2176.70 | 2176.12 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-09 09:15:00 | 2167.50 | 2174.86 | 2175.33 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 2182.20 | 2176.82 | 2176.15 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 10:15:00 | 2171.20 | 2177.10 | 2177.45 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 11:15:00 | 2195.00 | 2180.68 | 2179.05 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 2157.50 | 2179.86 | 2180.00 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2026-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 14:15:00 | 2180.00 | 2175.45 | 2175.11 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 09:15:00 | 2160.30 | 2173.15 | 2174.17 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 2187.50 | 2171.97 | 2171.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 13:15:00 | 2191.00 | 2178.15 | 2174.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 2170.90 | 2179.92 | 2176.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 2170.90 | 2179.92 | 2176.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 2170.90 | 2179.92 | 2176.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:00:00 | 2170.90 | 2179.92 | 2176.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 2186.10 | 2181.16 | 2177.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 13:00:00 | 2195.00 | 2184.12 | 2179.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 13:30:00 | 2198.90 | 2188.90 | 2181.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 09:15:00 | 2171.20 | 2189.82 | 2190.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 2171.20 | 2189.82 | 2190.78 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 14:15:00 | 2325.40 | 2215.83 | 2201.96 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 12:15:00 | 2170.20 | 2219.17 | 2222.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 13:15:00 | 2163.30 | 2208.00 | 2217.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 11:15:00 | 2183.00 | 2182.15 | 2198.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 12:00:00 | 2183.00 | 2182.15 | 2198.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 2208.90 | 2189.70 | 2198.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 15:00:00 | 2208.90 | 2189.70 | 2198.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 2203.00 | 2192.36 | 2198.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 2181.40 | 2192.36 | 2198.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 10:15:00 | 2072.33 | 2127.13 | 2148.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 12:15:00 | 2091.60 | 2089.93 | 2110.44 | SL hit (close>ema200) qty=0.50 sl=2089.93 alert=retest2 |

### Cycle 117 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 2121.20 | 2088.20 | 2087.76 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 10:15:00 | 2062.20 | 2089.25 | 2090.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 11:15:00 | 2038.00 | 2079.00 | 2085.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 10:15:00 | 2003.10 | 2000.83 | 2022.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-13 11:00:00 | 2003.10 | 2000.83 | 2022.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 2019.30 | 2005.51 | 2018.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 15:00:00 | 2019.30 | 2005.51 | 2018.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 2018.00 | 2008.01 | 2018.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 09:15:00 | 1957.70 | 2008.01 | 2018.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 13:15:00 | 1990.50 | 1980.62 | 1980.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 13:15:00 | 1990.50 | 1980.62 | 1980.13 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 1957.70 | 1978.76 | 1979.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1950.00 | 1964.40 | 1971.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 12:15:00 | 1976.00 | 1963.61 | 1967.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 12:15:00 | 1976.00 | 1963.61 | 1967.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 1976.00 | 1963.61 | 1967.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:45:00 | 1973.90 | 1963.61 | 1967.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 1971.20 | 1965.13 | 1968.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 15:00:00 | 1948.20 | 1961.74 | 1966.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-24 09:15:00 | 1850.79 | 1885.78 | 1917.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1885.60 | 1867.63 | 1889.55 | SL hit (close>ema200) qty=0.50 sl=1867.63 alert=retest2 |

### Cycle 121 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 1793.50 | 1718.44 | 1715.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 11:15:00 | 1802.80 | 1784.37 | 1768.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 1763.80 | 1786.71 | 1776.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 1763.80 | 1786.71 | 1776.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1763.80 | 1786.71 | 1776.64 | EMA400 retest candle locked (from upside) |

### Cycle 122 — SELL (started 2026-04-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 12:15:00 | 1742.90 | 1770.89 | 1771.29 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 1794.00 | 1770.03 | 1769.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 12:15:00 | 1809.00 | 1789.26 | 1780.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 13:15:00 | 1881.30 | 1886.21 | 1863.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 13:45:00 | 1878.00 | 1886.21 | 1863.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 1871.60 | 1885.05 | 1877.44 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2026-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 14:15:00 | 1847.30 | 1869.22 | 1872.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 09:15:00 | 1842.00 | 1860.86 | 1867.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 14:15:00 | 1848.30 | 1844.01 | 1854.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-23 15:00:00 | 1848.30 | 1844.01 | 1854.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 1818.60 | 1838.11 | 1850.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 10:45:00 | 1802.40 | 1829.87 | 1845.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 15:15:00 | 1841.40 | 1831.44 | 1830.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2026-04-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 15:15:00 | 1841.40 | 1831.44 | 1830.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 1845.00 | 1834.15 | 1831.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 11:15:00 | 1835.50 | 1836.65 | 1833.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 11:15:00 | 1835.50 | 1836.65 | 1833.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 1835.50 | 1836.65 | 1833.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:30:00 | 1829.70 | 1836.65 | 1833.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 1842.90 | 1837.90 | 1834.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:30:00 | 1837.90 | 1837.90 | 1834.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 1843.90 | 1839.10 | 1835.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:30:00 | 1838.70 | 1839.10 | 1835.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 1818.30 | 1834.94 | 1833.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 1818.30 | 1834.94 | 1833.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 1826.90 | 1833.33 | 1833.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 1842.90 | 1833.33 | 1833.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 09:15:00 | 1830.30 | 1832.73 | 1832.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 09:15:00 | 1830.30 | 1832.73 | 1832.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 13:15:00 | 1820.00 | 1828.42 | 1830.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 14:15:00 | 1818.50 | 1812.33 | 1818.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 14:15:00 | 1818.50 | 1812.33 | 1818.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 1818.50 | 1812.33 | 1818.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 15:00:00 | 1818.50 | 1812.33 | 1818.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 1828.00 | 1815.46 | 1819.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 1853.10 | 1815.46 | 1819.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1833.50 | 1819.07 | 1821.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:30:00 | 1839.00 | 1819.07 | 1821.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 1828.70 | 1823.39 | 1822.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 13:15:00 | 1837.40 | 1826.19 | 1824.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 1808.90 | 1828.34 | 1826.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 09:15:00 | 1808.90 | 1828.34 | 1826.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 1808.90 | 1828.34 | 1826.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:00:00 | 1808.90 | 1828.34 | 1826.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 1798.90 | 1822.45 | 1823.65 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 1836.40 | 1823.00 | 1822.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 15:15:00 | 1852.50 | 1828.90 | 1825.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 14:15:00 | 1833.70 | 1837.53 | 1832.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 14:15:00 | 1833.70 | 1837.53 | 1832.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 14:15:00 | 1833.70 | 1837.53 | 1832.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 15:00:00 | 1833.70 | 1837.53 | 1832.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1838.40 | 1838.34 | 1833.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 15:00:00 | 1855.20 | 1842.39 | 1837.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-11-13 09:15:00 | 1730.00 | 2024-11-19 11:15:00 | 1831.05 | STOP_HIT | 1.00 | -5.84% |
| BUY | retest2 | 2024-11-28 11:30:00 | 1901.20 | 2024-12-02 13:15:00 | 1885.40 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-12-12 09:15:00 | 1798.50 | 2024-12-16 14:15:00 | 1814.85 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-12-20 10:15:00 | 1759.50 | 2024-12-26 12:15:00 | 1779.45 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-12-23 09:15:00 | 1748.20 | 2024-12-26 12:15:00 | 1779.45 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2024-12-24 15:00:00 | 1764.30 | 2024-12-26 12:15:00 | 1779.45 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-12-26 09:15:00 | 1759.95 | 2024-12-26 12:15:00 | 1779.45 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-12-26 10:15:00 | 1751.05 | 2024-12-26 12:15:00 | 1779.45 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-01-02 12:15:00 | 1841.45 | 2025-01-06 14:15:00 | 1800.15 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-01-02 13:30:00 | 1825.00 | 2025-01-06 14:15:00 | 1800.15 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-01-03 09:15:00 | 1821.20 | 2025-01-06 14:15:00 | 1800.15 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-01-03 09:45:00 | 1821.75 | 2025-01-06 14:15:00 | 1800.15 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-01-13 09:15:00 | 1762.15 | 2025-01-14 10:15:00 | 1803.00 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-01-13 12:00:00 | 1771.05 | 2025-01-14 10:15:00 | 1803.00 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-01-17 10:30:00 | 1807.15 | 2025-01-17 15:15:00 | 1790.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-01-17 11:15:00 | 1802.15 | 2025-01-17 15:15:00 | 1790.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-01-17 12:45:00 | 1803.90 | 2025-01-17 15:15:00 | 1790.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-01-30 10:15:00 | 1644.60 | 2025-01-31 10:15:00 | 1661.70 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-02-07 10:45:00 | 1890.60 | 2025-02-11 13:15:00 | 1816.85 | STOP_HIT | 1.00 | -3.90% |
| SELL | retest2 | 2025-02-14 11:30:00 | 1802.00 | 2025-02-17 14:15:00 | 1851.55 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2025-02-28 12:15:00 | 1726.50 | 2025-03-06 10:15:00 | 1720.60 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2025-03-13 12:00:00 | 1659.75 | 2025-03-18 09:15:00 | 1576.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-13 15:00:00 | 1639.25 | 2025-03-18 10:15:00 | 1557.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-13 12:00:00 | 1659.75 | 2025-03-18 15:15:00 | 1582.00 | STOP_HIT | 0.50 | 4.68% |
| SELL | retest2 | 2025-03-13 15:00:00 | 1639.25 | 2025-03-18 15:15:00 | 1582.00 | STOP_HIT | 0.50 | 3.49% |
| BUY | retest2 | 2025-03-26 14:00:00 | 1700.50 | 2025-03-27 11:15:00 | 1700.00 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-03-27 09:45:00 | 1709.85 | 2025-03-27 11:15:00 | 1700.00 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-04-17 09:15:00 | 1648.00 | 2025-04-25 10:15:00 | 1668.70 | STOP_HIT | 1.00 | 1.26% |
| BUY | retest2 | 2025-05-07 10:15:00 | 1736.10 | 2025-05-09 09:15:00 | 1699.70 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-06-03 09:15:00 | 1844.10 | 2025-06-06 12:15:00 | 1855.10 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-06-13 10:15:00 | 1944.80 | 2025-06-18 10:15:00 | 1931.10 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-06-16 09:15:00 | 1959.30 | 2025-06-18 10:15:00 | 1931.10 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-06-16 10:15:00 | 1947.50 | 2025-06-18 10:15:00 | 1931.10 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-06-16 11:30:00 | 1947.70 | 2025-06-18 10:15:00 | 1931.10 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-06-16 13:45:00 | 1939.30 | 2025-06-18 10:15:00 | 1931.10 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-06-16 14:30:00 | 1940.80 | 2025-06-18 10:15:00 | 1931.10 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-06-17 09:15:00 | 1962.40 | 2025-06-18 10:15:00 | 1931.10 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-06-27 15:00:00 | 2187.90 | 2025-07-02 13:15:00 | 2141.70 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-07-02 13:00:00 | 2161.10 | 2025-07-02 13:15:00 | 2141.70 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-07-08 10:30:00 | 2060.50 | 2025-07-10 10:15:00 | 2090.30 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-07-09 09:30:00 | 2063.90 | 2025-07-10 10:15:00 | 2090.30 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-07-14 12:15:00 | 2136.60 | 2025-07-18 10:15:00 | 2125.00 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-07-15 12:00:00 | 2140.50 | 2025-07-18 10:15:00 | 2125.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-07-16 14:00:00 | 2133.40 | 2025-07-18 10:15:00 | 2125.00 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-07-17 10:15:00 | 2127.00 | 2025-07-18 10:15:00 | 2125.00 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2025-07-17 12:00:00 | 2138.20 | 2025-07-18 10:15:00 | 2125.00 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-07-28 09:15:00 | 2076.50 | 2025-07-30 14:15:00 | 2092.80 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-07-30 09:15:00 | 2080.00 | 2025-07-30 14:15:00 | 2092.80 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-08-29 09:30:00 | 2459.60 | 2025-08-29 12:15:00 | 2454.00 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-08-29 10:15:00 | 2465.00 | 2025-08-29 12:15:00 | 2454.00 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-09-15 13:15:00 | 2542.50 | 2025-09-19 14:15:00 | 2796.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-25 11:15:00 | 2764.50 | 2025-09-25 15:15:00 | 2715.00 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-09-29 11:30:00 | 2648.80 | 2025-10-01 11:15:00 | 2516.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-29 11:30:00 | 2648.80 | 2025-10-03 10:15:00 | 2556.70 | STOP_HIT | 0.50 | 3.48% |
| BUY | retest2 | 2025-10-16 09:15:00 | 2441.80 | 2025-10-16 10:15:00 | 2376.50 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-10-23 14:00:00 | 2299.00 | 2025-10-28 13:15:00 | 2306.50 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-11-10 15:15:00 | 2350.00 | 2025-11-12 09:15:00 | 2382.60 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-11-11 15:00:00 | 2355.10 | 2025-11-12 09:15:00 | 2382.60 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-11-24 09:15:00 | 2325.90 | 2025-11-28 10:15:00 | 2328.00 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-11-24 15:00:00 | 2291.00 | 2025-11-28 10:15:00 | 2328.00 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-11-25 10:45:00 | 2329.80 | 2025-11-28 10:15:00 | 2328.00 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2025-12-02 13:15:00 | 2378.50 | 2025-12-03 09:15:00 | 2341.00 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-12-03 12:45:00 | 2375.90 | 2025-12-05 09:15:00 | 2347.00 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-12-04 09:15:00 | 2386.60 | 2025-12-05 09:15:00 | 2347.00 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-12-04 10:15:00 | 2385.10 | 2025-12-05 09:15:00 | 2347.00 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-12-18 09:15:00 | 2266.80 | 2025-12-19 11:15:00 | 2309.10 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-12-18 10:45:00 | 2269.70 | 2025-12-19 11:15:00 | 2309.10 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-12-26 15:00:00 | 2324.30 | 2025-12-29 09:15:00 | 2302.60 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2026-01-01 10:15:00 | 2286.80 | 2026-01-06 11:15:00 | 2295.90 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2026-01-01 11:30:00 | 2290.00 | 2026-01-06 11:15:00 | 2295.90 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2026-01-01 12:00:00 | 2290.00 | 2026-01-06 11:15:00 | 2295.90 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2026-01-02 09:15:00 | 2267.90 | 2026-01-06 11:15:00 | 2295.90 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2026-01-19 10:45:00 | 2345.50 | 2026-01-20 14:15:00 | 2316.00 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2026-01-19 12:15:00 | 2350.10 | 2026-01-20 14:15:00 | 2316.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2026-01-19 13:30:00 | 2345.70 | 2026-01-20 14:15:00 | 2316.00 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2026-01-19 14:30:00 | 2346.20 | 2026-01-20 14:15:00 | 2316.00 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2026-01-23 11:30:00 | 2246.50 | 2026-01-28 09:15:00 | 2134.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 11:30:00 | 2246.50 | 2026-01-29 14:15:00 | 2150.20 | STOP_HIT | 0.50 | 4.29% |
| SELL | retest2 | 2026-01-27 09:15:00 | 2185.00 | 2026-02-01 09:15:00 | 2209.80 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2026-02-04 13:45:00 | 2190.60 | 2026-02-05 09:15:00 | 2151.20 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2026-02-04 15:00:00 | 2190.60 | 2026-02-05 09:15:00 | 2151.20 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2026-02-18 13:00:00 | 2195.00 | 2026-02-20 09:15:00 | 2171.20 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2026-02-18 13:30:00 | 2198.90 | 2026-02-20 09:15:00 | 2171.20 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-02-26 09:15:00 | 2181.40 | 2026-03-04 10:15:00 | 2072.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 09:15:00 | 2181.40 | 2026-03-05 12:15:00 | 2091.60 | STOP_HIT | 0.50 | 4.12% |
| SELL | retest2 | 2026-03-16 09:15:00 | 1957.70 | 2026-03-18 13:15:00 | 1990.50 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-03-20 15:00:00 | 1948.20 | 2026-03-24 09:15:00 | 1850.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 15:00:00 | 1948.20 | 2026-03-25 09:15:00 | 1885.60 | STOP_HIT | 0.50 | 3.21% |
| SELL | retest2 | 2026-04-24 10:45:00 | 1802.40 | 2026-04-27 15:15:00 | 1841.40 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2026-04-29 09:15:00 | 1842.90 | 2026-04-29 09:15:00 | 1830.30 | STOP_HIT | 1.00 | -0.68% |
