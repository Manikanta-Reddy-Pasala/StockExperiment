# Coforge Ltd. (COFORGE)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1365.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 68 |
| ALERT1 | 49 |
| ALERT2 | 46 |
| ALERT2_SKIP | 20 |
| ALERT3 | 115 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 53 |
| PARTIAL | 8 |
| TARGET_HIT | 9 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 62 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 36
- **Target hits / Stop hits / Partials:** 6 / 48 / 8
- **Avg / median % per leg:** 0.92% / -0.65%
- **Sum % (uncompounded):** 57.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 27 | 6 | 22.2% | 0 | 27 | 0 | -0.83% | -22.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 27 | 6 | 22.2% | 0 | 27 | 0 | -0.83% | -22.4% |
| SELL (all) | 35 | 20 | 57.1% | 6 | 21 | 8 | 2.27% | 79.6% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.52% | -1.5% |
| SELL @ 3rd Alert (retest2) | 34 | 20 | 58.8% | 6 | 20 | 8 | 2.39% | 81.1% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.52% | -1.5% |
| retest2 (combined) | 61 | 26 | 42.6% | 6 | 47 | 8 | 0.96% | 58.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 12:15:00 | 1658.50 | 1672.76 | 1673.84 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 10:15:00 | 1682.70 | 1673.09 | 1672.71 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-05-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 11:15:00 | 1666.30 | 1671.73 | 1672.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 12:15:00 | 1663.70 | 1670.13 | 1671.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 13:15:00 | 1654.50 | 1653.97 | 1660.04 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-22 09:15:00 | 1643.80 | 1653.59 | 1658.81 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 1668.80 | 1656.63 | 1659.72 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-05-22 09:15:00 | 1668.80 | 1656.63 | 1659.72 | SL hit (close>ema400) qty=1.00 sl=1659.72 alert=retest1 |

### Cycle 4 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 1689.70 | 1659.78 | 1658.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 1700.50 | 1692.26 | 1685.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 12:15:00 | 1693.20 | 1693.83 | 1688.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 13:00:00 | 1693.20 | 1693.83 | 1688.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1700.00 | 1712.14 | 1703.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:45:00 | 1704.20 | 1712.14 | 1703.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 1709.00 | 1711.51 | 1703.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 11:15:00 | 1712.10 | 1711.51 | 1703.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 12:45:00 | 1711.90 | 1710.79 | 1704.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 13:30:00 | 1711.80 | 1710.67 | 1705.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 10:45:00 | 1715.70 | 1710.39 | 1706.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 1713.60 | 1712.33 | 1708.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 13:00:00 | 1713.60 | 1712.33 | 1708.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 1720.00 | 1716.19 | 1711.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:45:00 | 1709.20 | 1714.65 | 1711.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 1711.90 | 1714.10 | 1711.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:15:00 | 1707.80 | 1714.10 | 1711.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 11:15:00 | 1706.70 | 1712.62 | 1710.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 12:15:00 | 1707.30 | 1712.62 | 1710.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 1703.50 | 1710.80 | 1710.06 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-03 14:15:00 | 1700.60 | 1707.94 | 1708.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 1700.60 | 1707.94 | 1708.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 15:15:00 | 1698.70 | 1706.09 | 1707.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 1720.00 | 1708.87 | 1709.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 1720.00 | 1708.87 | 1709.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 1720.00 | 1708.87 | 1709.01 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 10:15:00 | 1719.50 | 1711.00 | 1709.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 1746.00 | 1724.29 | 1717.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 10:15:00 | 1820.00 | 1826.30 | 1809.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 10:30:00 | 1821.50 | 1826.30 | 1809.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 1816.50 | 1822.11 | 1812.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:30:00 | 1814.50 | 1822.11 | 1812.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 1779.50 | 1813.09 | 1810.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:45:00 | 1774.50 | 1813.09 | 1810.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 1792.50 | 1808.97 | 1808.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 11:30:00 | 1800.00 | 1808.98 | 1808.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 12:00:00 | 1809.00 | 1808.98 | 1808.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 1788.50 | 1805.68 | 1807.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 1788.50 | 1805.68 | 1807.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 15:15:00 | 1783.00 | 1798.08 | 1803.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 10:15:00 | 1801.50 | 1798.67 | 1802.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 10:15:00 | 1801.50 | 1798.67 | 1802.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 1801.50 | 1798.67 | 1802.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 10:30:00 | 1808.00 | 1798.67 | 1802.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 11:15:00 | 1805.00 | 1799.94 | 1802.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 11:30:00 | 1802.50 | 1799.94 | 1802.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 12:15:00 | 1791.50 | 1798.25 | 1801.86 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 1836.50 | 1808.68 | 1805.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 12:15:00 | 1846.50 | 1816.24 | 1808.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 15:15:00 | 1834.50 | 1837.40 | 1827.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 09:15:00 | 1840.50 | 1837.40 | 1827.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 1832.00 | 1836.32 | 1828.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:15:00 | 1826.00 | 1836.32 | 1828.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 1815.50 | 1832.16 | 1826.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:45:00 | 1815.00 | 1832.16 | 1826.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 1814.00 | 1828.53 | 1825.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:30:00 | 1819.50 | 1828.53 | 1825.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 09:15:00 | 1807.00 | 1825.27 | 1825.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 10:15:00 | 1787.00 | 1817.62 | 1821.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 1805.50 | 1800.53 | 1809.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 10:15:00 | 1805.50 | 1800.53 | 1809.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1805.50 | 1800.53 | 1809.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:45:00 | 1807.50 | 1800.53 | 1809.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1818.50 | 1804.12 | 1809.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:00:00 | 1818.50 | 1804.12 | 1809.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 1815.00 | 1806.30 | 1810.42 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 1831.50 | 1813.69 | 1813.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 1835.00 | 1823.49 | 1818.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 1865.50 | 1874.08 | 1861.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 09:15:00 | 1865.50 | 1874.08 | 1861.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 1865.50 | 1874.08 | 1861.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:30:00 | 1863.00 | 1874.08 | 1861.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 1903.00 | 1906.77 | 1893.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 1890.00 | 1906.77 | 1893.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1901.00 | 1905.62 | 1894.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 13:30:00 | 1918.00 | 1912.36 | 1901.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 14:00:00 | 1923.10 | 1923.79 | 1914.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 13:45:00 | 1919.50 | 1924.51 | 1919.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 12:15:00 | 1936.40 | 1947.67 | 1947.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2025-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 12:15:00 | 1936.40 | 1947.67 | 1947.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 13:15:00 | 1926.90 | 1943.52 | 1945.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 1877.90 | 1874.05 | 1891.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 10:00:00 | 1877.90 | 1874.05 | 1891.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 1872.00 | 1873.64 | 1890.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:15:00 | 1868.30 | 1873.64 | 1890.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 14:30:00 | 1869.60 | 1870.17 | 1882.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 1891.80 | 1875.11 | 1882.88 | SL hit (close>static) qty=1.00 sl=1890.80 alert=retest2 |

### Cycle 12 — BUY (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 14:15:00 | 1895.60 | 1887.35 | 1887.04 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 09:15:00 | 1875.50 | 1885.76 | 1886.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 09:15:00 | 1872.00 | 1880.44 | 1883.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 11:15:00 | 1874.50 | 1864.47 | 1871.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 11:15:00 | 1874.50 | 1864.47 | 1871.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 1874.50 | 1864.47 | 1871.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:45:00 | 1874.80 | 1864.47 | 1871.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 1861.80 | 1863.94 | 1870.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:30:00 | 1871.00 | 1863.94 | 1870.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 1873.30 | 1864.31 | 1868.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:30:00 | 1866.80 | 1864.31 | 1868.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1877.90 | 1867.03 | 1868.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:00:00 | 1877.90 | 1867.03 | 1868.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 1869.10 | 1867.44 | 1868.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 13:45:00 | 1866.70 | 1867.26 | 1868.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:45:00 | 1861.60 | 1867.50 | 1868.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:30:00 | 1861.00 | 1867.36 | 1868.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 09:15:00 | 1842.50 | 1861.50 | 1864.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1850.30 | 1859.26 | 1863.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 13:15:00 | 1840.90 | 1855.40 | 1860.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 1760.70 | 1851.91 | 1857.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 09:15:00 | 1773.37 | 1821.25 | 1842.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 09:15:00 | 1768.52 | 1821.25 | 1842.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 09:15:00 | 1767.95 | 1821.25 | 1842.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 09:15:00 | 1750.38 | 1821.25 | 1842.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 09:15:00 | 1748.86 | 1821.25 | 1842.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-07-24 14:15:00 | 1680.03 | 1732.99 | 1785.38 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 14 — BUY (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 12:15:00 | 1740.20 | 1726.02 | 1724.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 13:15:00 | 1743.80 | 1729.58 | 1726.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 1707.40 | 1730.63 | 1728.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 1707.40 | 1730.63 | 1728.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1707.40 | 1730.63 | 1728.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:00:00 | 1707.40 | 1730.63 | 1728.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 1715.60 | 1727.62 | 1726.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:30:00 | 1725.50 | 1727.64 | 1726.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 14:15:00 | 1702.70 | 1726.38 | 1729.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 1702.70 | 1726.38 | 1729.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 10:15:00 | 1701.10 | 1716.48 | 1723.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 1724.10 | 1717.69 | 1722.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 12:15:00 | 1724.10 | 1717.69 | 1722.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 1724.10 | 1717.69 | 1722.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 1724.10 | 1717.69 | 1722.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 1742.00 | 1722.55 | 1724.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 1742.00 | 1722.55 | 1724.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 1747.80 | 1727.60 | 1726.73 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 11:15:00 | 1721.40 | 1726.33 | 1726.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 12:15:00 | 1704.00 | 1721.86 | 1724.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 09:15:00 | 1673.80 | 1661.99 | 1683.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 10:00:00 | 1673.80 | 1661.99 | 1683.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 1681.60 | 1665.58 | 1681.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 12:00:00 | 1681.60 | 1665.58 | 1681.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 1673.90 | 1667.24 | 1680.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 13:15:00 | 1669.60 | 1667.24 | 1680.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 13:15:00 | 1685.90 | 1670.97 | 1680.97 | SL hit (close>static) qty=1.00 sl=1684.70 alert=retest2 |

### Cycle 18 — BUY (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 11:15:00 | 1640.10 | 1628.07 | 1627.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 1661.00 | 1639.85 | 1633.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 1646.90 | 1649.21 | 1642.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 1646.90 | 1649.21 | 1642.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 1646.90 | 1649.21 | 1642.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:45:00 | 1647.20 | 1649.21 | 1642.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 1651.40 | 1651.06 | 1646.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 10:30:00 | 1663.70 | 1654.37 | 1648.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 10:15:00 | 1730.10 | 1743.84 | 1744.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2025-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 10:15:00 | 1730.10 | 1743.84 | 1744.50 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 11:15:00 | 1754.70 | 1743.73 | 1743.00 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2025-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 13:15:00 | 1734.10 | 1741.37 | 1742.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 14:15:00 | 1722.20 | 1737.53 | 1740.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 1751.20 | 1738.42 | 1740.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 1751.20 | 1738.42 | 1740.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1751.20 | 1738.42 | 1740.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:30:00 | 1763.20 | 1738.42 | 1740.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 1759.00 | 1742.54 | 1741.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 1763.30 | 1746.69 | 1743.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 1754.20 | 1763.31 | 1757.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 1754.20 | 1763.31 | 1757.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 1754.20 | 1763.31 | 1757.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 1754.20 | 1763.31 | 1757.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 1755.40 | 1761.73 | 1756.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:30:00 | 1757.80 | 1761.73 | 1756.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 1754.00 | 1760.18 | 1756.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 1753.40 | 1760.18 | 1756.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 10:15:00 | 1729.40 | 1751.12 | 1752.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 12:15:00 | 1725.10 | 1742.09 | 1748.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 10:15:00 | 1676.00 | 1669.12 | 1687.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 11:00:00 | 1676.00 | 1669.12 | 1687.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 1677.50 | 1671.64 | 1685.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:30:00 | 1680.20 | 1671.64 | 1685.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 1693.20 | 1672.17 | 1680.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:45:00 | 1694.80 | 1672.17 | 1680.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 1696.30 | 1677.00 | 1682.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:00:00 | 1696.30 | 1677.00 | 1682.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2025-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 12:15:00 | 1700.00 | 1686.19 | 1685.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 1754.20 | 1702.50 | 1693.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 14:15:00 | 1761.80 | 1767.86 | 1755.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 15:00:00 | 1761.80 | 1767.86 | 1755.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1743.80 | 1763.23 | 1755.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:00:00 | 1743.80 | 1763.23 | 1755.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 1748.00 | 1760.18 | 1755.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:30:00 | 1746.20 | 1760.18 | 1755.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 1752.50 | 1756.02 | 1754.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:30:00 | 1750.00 | 1756.02 | 1754.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 1759.00 | 1755.66 | 1754.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 1760.80 | 1755.66 | 1754.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:45:00 | 1765.20 | 1757.52 | 1755.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 1732.60 | 1789.38 | 1793.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 1732.60 | 1789.38 | 1793.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 1687.80 | 1729.17 | 1755.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 14:15:00 | 1553.00 | 1548.25 | 1572.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 14:45:00 | 1555.10 | 1548.25 | 1572.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 1601.10 | 1558.62 | 1573.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:00:00 | 1601.10 | 1558.62 | 1573.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 1606.30 | 1568.16 | 1576.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 11:00:00 | 1606.30 | 1568.16 | 1576.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2025-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 14:15:00 | 1591.60 | 1582.45 | 1581.61 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 09:15:00 | 1563.30 | 1579.83 | 1580.64 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 1586.10 | 1580.88 | 1580.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 1600.10 | 1587.07 | 1583.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 12:15:00 | 1649.90 | 1650.35 | 1633.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 13:00:00 | 1649.90 | 1650.35 | 1633.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1713.00 | 1721.86 | 1712.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 09:15:00 | 1726.50 | 1714.18 | 1711.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 11:15:00 | 1690.20 | 1707.43 | 1708.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 1690.20 | 1707.43 | 1708.82 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 1750.00 | 1712.87 | 1710.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 11:15:00 | 1758.50 | 1728.37 | 1718.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 1728.20 | 1751.30 | 1743.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 1728.20 | 1751.30 | 1743.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 1728.20 | 1751.30 | 1743.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 1725.00 | 1751.30 | 1743.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 1734.10 | 1747.86 | 1742.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:30:00 | 1722.50 | 1747.86 | 1742.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 1730.00 | 1740.36 | 1740.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:45:00 | 1726.20 | 1740.36 | 1740.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — SELL (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 14:15:00 | 1733.10 | 1738.91 | 1739.54 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 1748.20 | 1740.59 | 1740.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 1773.60 | 1749.42 | 1745.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 1751.60 | 1759.03 | 1752.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 14:15:00 | 1751.60 | 1759.03 | 1752.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1751.60 | 1759.03 | 1752.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 1751.60 | 1759.03 | 1752.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 1761.70 | 1759.57 | 1753.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 1769.80 | 1759.57 | 1753.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 11:30:00 | 1764.50 | 1762.44 | 1756.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 1832.50 | 1761.05 | 1757.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 11:15:00 | 1793.50 | 1801.38 | 1801.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 11:15:00 | 1793.50 | 1801.38 | 1801.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 1775.30 | 1794.52 | 1798.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 13:15:00 | 1787.70 | 1780.78 | 1788.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 13:15:00 | 1787.70 | 1780.78 | 1788.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 1787.70 | 1780.78 | 1788.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:00:00 | 1787.70 | 1780.78 | 1788.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 1796.40 | 1783.90 | 1788.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 1796.40 | 1783.90 | 1788.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 1794.90 | 1786.10 | 1789.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 1775.00 | 1786.10 | 1789.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 1770.80 | 1758.24 | 1758.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 09:15:00 | 1770.80 | 1758.24 | 1758.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 10:15:00 | 1779.80 | 1762.55 | 1760.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 1809.70 | 1818.33 | 1806.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 15:00:00 | 1809.70 | 1818.33 | 1806.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 1810.30 | 1816.73 | 1807.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 1791.30 | 1816.73 | 1807.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1778.90 | 1809.16 | 1804.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:00:00 | 1778.90 | 1809.16 | 1804.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 1782.10 | 1803.75 | 1802.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:30:00 | 1774.60 | 1803.75 | 1802.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 1775.40 | 1798.08 | 1800.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 1760.20 | 1790.50 | 1796.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 14:15:00 | 1802.60 | 1790.92 | 1795.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 14:15:00 | 1802.60 | 1790.92 | 1795.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 1802.60 | 1790.92 | 1795.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 15:00:00 | 1802.60 | 1790.92 | 1795.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 1802.00 | 1793.14 | 1796.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 1809.20 | 1793.14 | 1796.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 1796.80 | 1796.48 | 1797.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:45:00 | 1797.40 | 1796.48 | 1797.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 12:15:00 | 1802.60 | 1797.70 | 1797.66 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 1777.50 | 1795.48 | 1796.91 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 10:15:00 | 1823.00 | 1798.50 | 1795.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 11:15:00 | 1839.70 | 1806.74 | 1799.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 14:15:00 | 1849.00 | 1850.20 | 1834.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-20 15:00:00 | 1849.00 | 1850.20 | 1834.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1797.40 | 1839.45 | 1831.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 1797.40 | 1839.45 | 1831.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 1800.60 | 1831.68 | 1829.08 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 1803.30 | 1826.00 | 1826.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 15:15:00 | 1794.90 | 1808.94 | 1817.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 1824.80 | 1812.11 | 1818.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 1824.80 | 1812.11 | 1818.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 1824.80 | 1812.11 | 1818.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:30:00 | 1825.50 | 1812.11 | 1818.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 1818.60 | 1813.41 | 1818.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 11:45:00 | 1813.00 | 1812.45 | 1817.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 14:15:00 | 1832.10 | 1814.86 | 1814.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2025-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 14:15:00 | 1832.10 | 1814.86 | 1814.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 1855.00 | 1825.31 | 1819.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 14:15:00 | 1908.40 | 1909.55 | 1890.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 15:00:00 | 1908.40 | 1909.55 | 1890.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 1903.90 | 1912.07 | 1902.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:45:00 | 1899.30 | 1912.07 | 1902.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 1902.00 | 1910.06 | 1902.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 1920.90 | 1910.06 | 1902.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 13:45:00 | 1910.90 | 1911.12 | 1905.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 15:00:00 | 1913.40 | 1911.57 | 1906.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 11:15:00 | 1923.60 | 1909.54 | 1906.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 1919.60 | 1911.55 | 1908.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 09:15:00 | 1945.30 | 1913.42 | 1910.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 09:15:00 | 1872.70 | 1938.07 | 1946.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 1872.70 | 1938.07 | 1946.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 10:15:00 | 1857.90 | 1881.43 | 1906.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 14:15:00 | 1840.20 | 1839.70 | 1860.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 15:00:00 | 1840.20 | 1839.70 | 1860.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 1852.10 | 1842.17 | 1851.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 15:00:00 | 1852.10 | 1842.17 | 1851.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 15:15:00 | 1852.20 | 1844.17 | 1851.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:15:00 | 1851.20 | 1844.17 | 1851.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1848.80 | 1845.10 | 1851.15 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 12:15:00 | 1867.50 | 1855.65 | 1855.00 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 13:15:00 | 1850.70 | 1856.54 | 1856.78 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 14:15:00 | 1865.90 | 1858.41 | 1857.61 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2025-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 13:15:00 | 1850.20 | 1856.93 | 1857.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 14:15:00 | 1843.10 | 1854.16 | 1856.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 1850.10 | 1848.87 | 1852.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 11:45:00 | 1850.00 | 1848.87 | 1852.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 1844.70 | 1847.08 | 1851.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:30:00 | 1850.00 | 1847.08 | 1851.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 1855.50 | 1848.77 | 1851.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:45:00 | 1854.00 | 1848.77 | 1851.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 1853.00 | 1849.61 | 1851.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 1848.00 | 1849.61 | 1851.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1844.60 | 1848.61 | 1851.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:00:00 | 1842.60 | 1848.05 | 1850.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:45:00 | 1842.10 | 1847.08 | 1849.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 13:30:00 | 1842.40 | 1847.00 | 1849.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 15:15:00 | 1840.00 | 1846.74 | 1849.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 1865.70 | 1849.45 | 1849.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:00:00 | 1865.70 | 1849.45 | 1849.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-22 10:15:00 | 1871.20 | 1853.80 | 1851.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 1871.20 | 1853.80 | 1851.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 11:15:00 | 1873.10 | 1857.66 | 1853.75 | Break + close above crossover candle high |

### Cycle 47 — SELL (started 2025-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 09:15:00 | 1797.70 | 1850.28 | 1852.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 1738.10 | 1761.99 | 1790.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 13:15:00 | 1685.00 | 1683.03 | 1713.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 14:00:00 | 1685.00 | 1683.03 | 1713.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1661.50 | 1661.31 | 1670.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 14:15:00 | 1657.50 | 1662.60 | 1668.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 15:15:00 | 1657.10 | 1662.12 | 1667.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 15:15:00 | 1655.80 | 1650.91 | 1650.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2026-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 15:15:00 | 1655.80 | 1650.91 | 1650.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 09:15:00 | 1693.90 | 1659.51 | 1654.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 1678.20 | 1685.44 | 1674.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 11:00:00 | 1678.20 | 1685.44 | 1674.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 1672.00 | 1682.75 | 1674.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 1672.00 | 1682.75 | 1674.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 1671.20 | 1680.44 | 1673.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:00:00 | 1671.20 | 1680.44 | 1673.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 1649.40 | 1674.23 | 1671.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 14:00:00 | 1649.40 | 1674.23 | 1671.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2026-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 14:15:00 | 1648.50 | 1669.09 | 1669.49 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 11:15:00 | 1675.20 | 1669.29 | 1669.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-09 13:15:00 | 1682.10 | 1672.81 | 1670.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 09:15:00 | 1696.20 | 1700.36 | 1692.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-14 10:00:00 | 1696.20 | 1700.36 | 1692.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 1699.80 | 1700.25 | 1693.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 1723.20 | 1691.70 | 1691.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 10:30:00 | 1709.90 | 1720.35 | 1717.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 12:15:00 | 1699.80 | 1713.13 | 1714.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 1699.80 | 1713.13 | 1714.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 1689.10 | 1708.32 | 1712.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 14:15:00 | 1668.80 | 1659.72 | 1679.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 15:00:00 | 1668.80 | 1659.72 | 1679.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1714.90 | 1671.90 | 1681.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 1723.10 | 1671.90 | 1681.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1702.00 | 1677.92 | 1683.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:30:00 | 1683.60 | 1678.34 | 1683.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 10:15:00 | 1674.90 | 1665.64 | 1664.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 1674.90 | 1665.64 | 1664.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 13:15:00 | 1688.00 | 1673.40 | 1668.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 1664.50 | 1676.29 | 1671.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 1664.50 | 1676.29 | 1671.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 1664.50 | 1676.29 | 1671.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 1664.50 | 1676.29 | 1671.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 1668.50 | 1674.73 | 1671.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 13:45:00 | 1672.00 | 1672.82 | 1671.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 10:15:00 | 1664.30 | 1670.91 | 1670.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 10:15:00 | 1664.30 | 1670.91 | 1670.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 14:15:00 | 1653.20 | 1664.02 | 1667.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 11:15:00 | 1658.20 | 1655.35 | 1661.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 1658.20 | 1655.35 | 1661.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1658.20 | 1655.35 | 1661.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 1658.20 | 1655.35 | 1661.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1660.80 | 1656.44 | 1661.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:00:00 | 1660.80 | 1656.44 | 1661.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 1682.60 | 1661.67 | 1663.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 1682.60 | 1661.67 | 1663.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 1659.30 | 1661.20 | 1662.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 15:15:00 | 1648.00 | 1661.20 | 1662.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 10:00:00 | 1656.30 | 1658.11 | 1661.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1706.10 | 1668.58 | 1663.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1706.10 | 1668.58 | 1663.91 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2026-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 10:15:00 | 1590.40 | 1665.34 | 1671.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 11:15:00 | 1580.80 | 1648.43 | 1663.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 1554.70 | 1553.44 | 1577.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 11:00:00 | 1554.70 | 1553.44 | 1577.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1557.00 | 1552.40 | 1566.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:30:00 | 1563.60 | 1552.40 | 1566.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 1573.50 | 1557.21 | 1566.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 12:00:00 | 1573.50 | 1557.21 | 1566.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 1573.60 | 1560.49 | 1566.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 13:00:00 | 1573.60 | 1560.49 | 1566.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 1425.80 | 1388.96 | 1404.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 1429.50 | 1388.96 | 1404.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 1404.60 | 1392.09 | 1404.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 13:45:00 | 1395.70 | 1396.92 | 1403.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 09:15:00 | 1325.91 | 1369.48 | 1376.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-24 09:15:00 | 1256.13 | 1286.85 | 1319.14 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 56 — BUY (started 2026-03-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 15:15:00 | 1167.90 | 1162.41 | 1161.89 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2026-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 09:15:00 | 1155.60 | 1161.05 | 1161.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-10 11:15:00 | 1140.70 | 1156.54 | 1159.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 12:15:00 | 1121.80 | 1114.69 | 1126.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 13:00:00 | 1121.80 | 1114.69 | 1126.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 1114.50 | 1114.65 | 1125.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 14:15:00 | 1108.60 | 1114.65 | 1125.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 09:15:00 | 1053.17 | 1073.35 | 1087.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-17 12:15:00 | 1074.90 | 1069.62 | 1081.91 | SL hit (close>ema200) qty=0.50 sl=1069.62 alert=retest2 |

### Cycle 58 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 1137.30 | 1093.88 | 1089.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 1140.00 | 1115.96 | 1101.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1112.00 | 1119.50 | 1107.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1112.00 | 1119.50 | 1107.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1112.00 | 1119.50 | 1107.16 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2026-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 15:15:00 | 1096.90 | 1101.51 | 1102.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 14:15:00 | 1084.60 | 1094.84 | 1098.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 15:15:00 | 1098.00 | 1095.47 | 1098.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 15:15:00 | 1098.00 | 1095.47 | 1098.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 1098.00 | 1095.47 | 1098.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 1077.90 | 1095.47 | 1098.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:45:00 | 1081.40 | 1092.58 | 1096.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 13:15:00 | 1101.80 | 1095.38 | 1096.82 | SL hit (close>static) qty=1.00 sl=1100.00 alert=retest2 |

### Cycle 60 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 1122.90 | 1102.16 | 1099.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 1127.50 | 1111.62 | 1105.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 13:15:00 | 1148.10 | 1152.99 | 1138.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 14:00:00 | 1148.10 | 1152.99 | 1138.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 1120.60 | 1144.34 | 1138.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:00:00 | 1120.60 | 1144.34 | 1138.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 13:15:00 | 1122.30 | 1132.69 | 1133.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 1116.00 | 1129.35 | 1132.28 | Break + close below crossover candle low |

### Cycle 62 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 1182.20 | 1137.48 | 1135.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 12:15:00 | 1193.90 | 1169.70 | 1156.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 1242.10 | 1252.96 | 1236.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 1242.10 | 1252.96 | 1236.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 1227.00 | 1253.60 | 1245.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 10:00:00 | 1227.00 | 1253.60 | 1245.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 1220.60 | 1247.00 | 1243.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 11:15:00 | 1215.80 | 1247.00 | 1243.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2026-04-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-10 12:15:00 | 1225.30 | 1238.02 | 1239.67 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 1257.00 | 1235.47 | 1234.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 13:15:00 | 1285.50 | 1257.38 | 1246.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 12:15:00 | 1308.70 | 1309.36 | 1292.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-17 12:45:00 | 1304.70 | 1309.36 | 1292.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 1295.70 | 1309.26 | 1297.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 10:00:00 | 1295.70 | 1309.26 | 1297.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 1303.20 | 1308.05 | 1298.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 11:15:00 | 1307.60 | 1308.05 | 1298.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 12:00:00 | 1307.80 | 1308.00 | 1299.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 13:15:00 | 1289.40 | 1302.76 | 1298.26 | SL hit (close<static) qty=1.00 sl=1291.00 alert=retest2 |

### Cycle 65 — SELL (started 2026-04-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 14:15:00 | 1292.20 | 1295.70 | 1296.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 1240.30 | 1284.41 | 1290.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1186.90 | 1171.45 | 1199.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:45:00 | 1187.40 | 1171.45 | 1199.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 1201.80 | 1183.70 | 1197.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 14:00:00 | 1201.80 | 1183.70 | 1197.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 1204.20 | 1187.80 | 1197.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 15:00:00 | 1204.20 | 1187.80 | 1197.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 1202.30 | 1194.58 | 1198.71 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 15:15:00 | 1206.80 | 1201.03 | 1200.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 1222.30 | 1205.28 | 1202.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 1207.10 | 1208.54 | 1205.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 13:15:00 | 1207.10 | 1208.54 | 1205.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 1207.10 | 1208.54 | 1205.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 1207.10 | 1208.54 | 1205.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 1204.10 | 1207.65 | 1205.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:00:00 | 1204.10 | 1207.65 | 1205.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 1202.50 | 1206.62 | 1205.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 1181.80 | 1206.62 | 1205.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 1191.00 | 1201.77 | 1203.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 09:15:00 | 1172.00 | 1193.15 | 1198.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 12:15:00 | 1165.90 | 1163.58 | 1174.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-05 13:00:00 | 1165.90 | 1163.58 | 1174.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 68 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 1264.00 | 1185.49 | 1181.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 09:15:00 | 1337.30 | 1291.04 | 1261.65 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-22 09:15:00 | 1643.80 | 2025-05-22 09:15:00 | 1668.80 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-05-22 12:15:00 | 1638.90 | 2025-05-23 09:15:00 | 1689.70 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest2 | 2025-05-30 11:15:00 | 1712.10 | 2025-06-03 14:15:00 | 1700.60 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-05-30 12:45:00 | 1711.90 | 2025-06-03 14:15:00 | 1700.60 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-05-30 13:30:00 | 1711.80 | 2025-06-03 14:15:00 | 1700.60 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-06-02 10:45:00 | 1715.70 | 2025-06-03 14:15:00 | 1700.60 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-06-12 11:30:00 | 1800.00 | 2025-06-12 13:15:00 | 1788.50 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-06-12 12:00:00 | 1809.00 | 2025-06-12 13:15:00 | 1788.50 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-06-30 13:30:00 | 1918.00 | 2025-07-09 12:15:00 | 1936.40 | STOP_HIT | 1.00 | 0.96% |
| BUY | retest2 | 2025-07-01 14:00:00 | 1923.10 | 2025-07-09 12:15:00 | 1936.40 | STOP_HIT | 1.00 | 0.69% |
| BUY | retest2 | 2025-07-02 13:45:00 | 1919.50 | 2025-07-09 12:15:00 | 1936.40 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2025-07-14 11:15:00 | 1868.30 | 2025-07-15 09:15:00 | 1891.80 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-07-14 14:30:00 | 1869.60 | 2025-07-15 09:15:00 | 1891.80 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-07-21 13:45:00 | 1866.70 | 2025-07-24 09:15:00 | 1773.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 09:45:00 | 1861.60 | 2025-07-24 09:15:00 | 1768.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 10:30:00 | 1861.00 | 2025-07-24 09:15:00 | 1767.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 09:15:00 | 1842.50 | 2025-07-24 09:15:00 | 1750.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 13:15:00 | 1840.90 | 2025-07-24 09:15:00 | 1748.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-21 13:45:00 | 1866.70 | 2025-07-24 14:15:00 | 1680.03 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-22 09:45:00 | 1861.60 | 2025-07-24 14:15:00 | 1675.44 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-22 10:30:00 | 1861.00 | 2025-07-24 14:15:00 | 1674.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-24 09:15:00 | 1760.70 | 2025-07-24 14:15:00 | 1672.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 09:15:00 | 1842.50 | 2025-07-25 09:15:00 | 1658.25 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-23 13:15:00 | 1840.90 | 2025-07-25 09:15:00 | 1656.81 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-24 09:15:00 | 1760.70 | 2025-07-28 09:15:00 | 1725.90 | STOP_HIT | 0.50 | 1.98% |
| BUY | retest2 | 2025-07-31 11:30:00 | 1725.50 | 2025-08-01 14:15:00 | 1702.70 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-08-07 13:15:00 | 1669.60 | 2025-08-07 13:15:00 | 1685.90 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-08-08 09:15:00 | 1636.70 | 2025-08-14 11:15:00 | 1640.10 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-08-20 10:30:00 | 1663.70 | 2025-08-28 10:15:00 | 1730.10 | STOP_HIT | 1.00 | 3.99% |
| BUY | retest2 | 2025-09-16 09:15:00 | 1760.80 | 2025-09-22 09:15:00 | 1732.60 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-09-16 09:45:00 | 1765.20 | 2025-09-22 09:15:00 | 1732.60 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-10-14 09:15:00 | 1726.50 | 2025-10-14 11:15:00 | 1690.20 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-10-24 09:15:00 | 1769.80 | 2025-10-31 11:15:00 | 1793.50 | STOP_HIT | 1.00 | 1.34% |
| BUY | retest2 | 2025-10-24 11:30:00 | 1764.50 | 2025-10-31 11:15:00 | 1793.50 | STOP_HIT | 1.00 | 1.64% |
| BUY | retest2 | 2025-10-27 09:15:00 | 1832.50 | 2025-10-31 11:15:00 | 1793.50 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-11-04 09:15:00 | 1775.00 | 2025-11-11 09:15:00 | 1770.80 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2025-11-24 11:45:00 | 1813.00 | 2025-11-25 14:15:00 | 1832.10 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-12-02 09:15:00 | 1920.90 | 2025-12-09 09:15:00 | 1872.70 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2025-12-02 13:45:00 | 1910.90 | 2025-12-09 09:15:00 | 1872.70 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-12-02 15:00:00 | 1913.40 | 2025-12-09 09:15:00 | 1872.70 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-12-03 11:15:00 | 1923.60 | 2025-12-09 09:15:00 | 1872.70 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2025-12-04 09:15:00 | 1945.30 | 2025-12-09 09:15:00 | 1872.70 | STOP_HIT | 1.00 | -3.73% |
| SELL | retest2 | 2025-12-19 12:00:00 | 1842.60 | 2025-12-22 10:15:00 | 1871.20 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-12-19 12:45:00 | 1842.10 | 2025-12-22 10:15:00 | 1871.20 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-12-19 13:30:00 | 1842.40 | 2025-12-22 10:15:00 | 1871.20 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-12-19 15:15:00 | 1840.00 | 2025-12-22 10:15:00 | 1871.20 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-01-01 14:15:00 | 1657.50 | 2026-01-06 15:15:00 | 1655.80 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2026-01-01 15:15:00 | 1657.10 | 2026-01-06 15:15:00 | 1655.80 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2026-01-16 09:15:00 | 1723.20 | 2026-01-20 12:15:00 | 1699.80 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2026-01-20 10:30:00 | 1709.90 | 2026-01-20 12:15:00 | 1699.80 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2026-01-22 11:30:00 | 1683.60 | 2026-01-28 10:15:00 | 1674.90 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2026-01-29 13:45:00 | 1672.00 | 2026-01-30 10:15:00 | 1664.30 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2026-02-01 15:15:00 | 1648.00 | 2026-02-03 09:15:00 | 1706.10 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2026-02-02 10:00:00 | 1656.30 | 2026-02-03 09:15:00 | 1706.10 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2026-02-17 13:45:00 | 1395.70 | 2026-02-20 09:15:00 | 1325.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 13:45:00 | 1395.70 | 2026-02-24 09:15:00 | 1256.13 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-12 14:15:00 | 1108.60 | 2026-03-17 09:15:00 | 1053.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 14:15:00 | 1108.60 | 2026-03-17 12:15:00 | 1074.90 | STOP_HIT | 0.50 | 3.04% |
| SELL | retest2 | 2026-03-23 09:15:00 | 1077.90 | 2026-03-23 13:15:00 | 1101.80 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2026-03-23 09:45:00 | 1081.40 | 2026-03-23 13:15:00 | 1101.80 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2026-04-20 11:15:00 | 1307.60 | 2026-04-20 13:15:00 | 1289.40 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2026-04-20 12:00:00 | 1307.80 | 2026-04-20 13:15:00 | 1289.40 | STOP_HIT | 1.00 | -1.41% |
