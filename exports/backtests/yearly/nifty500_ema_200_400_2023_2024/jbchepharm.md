# J.B. Chemicals & Pharmaceuticals Ltd. (JBCHEPHARM)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 2155.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 3 |
| ALERT3 | 59 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 33 |
| PARTIAL | 6 |
| TARGET_HIT | 7 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 39 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 20
- **Target hits / Stop hits / Partials:** 7 / 26 / 6
- **Avg / median % per leg:** 1.81% / -0.59%
- **Sum % (uncompounded):** 70.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 7 | 33.3% | 7 | 14 | 0 | 2.38% | 49.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 21 | 7 | 33.3% | 7 | 14 | 0 | 2.38% | 49.9% |
| SELL (all) | 18 | 12 | 66.7% | 0 | 12 | 6 | 1.15% | 20.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 12 | 66.7% | 0 | 12 | 6 | 1.15% | 20.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 39 | 19 | 48.7% | 7 | 26 | 6 | 1.81% | 70.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 11:15:00 | 1686.20 | 1873.99 | 1874.63 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-10-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-25 10:15:00 | 1890.70 | 1867.93 | 1867.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 10:15:00 | 1910.40 | 1869.03 | 1868.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-29 09:15:00 | 1856.85 | 1869.36 | 1868.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-29 09:15:00 | 1856.85 | 1869.36 | 1868.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 1856.85 | 1869.36 | 1868.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 10:00:00 | 1856.85 | 1869.36 | 1868.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 10:15:00 | 1857.05 | 1869.23 | 1868.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 10:30:00 | 1857.65 | 1869.23 | 1868.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 11:15:00 | 1871.30 | 1868.68 | 1868.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 12:00:00 | 1871.30 | 1868.68 | 1868.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 12:15:00 | 1865.65 | 1868.65 | 1868.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 12:45:00 | 1869.50 | 1868.65 | 1868.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 13:15:00 | 1863.15 | 1868.59 | 1868.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 13:45:00 | 1863.15 | 1868.59 | 1868.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 1874.00 | 1868.64 | 1868.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 09:15:00 | 1909.90 | 1868.71 | 1868.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 14:15:00 | 1852.70 | 1874.15 | 1871.22 | SL hit (close<static) qty=1.00 sl=1860.10 alert=retest2 |

### Cycle 3 — SELL (started 2024-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 10:15:00 | 1814.95 | 1868.47 | 1868.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 1737.00 | 1857.68 | 1863.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 1810.80 | 1805.03 | 1832.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-25 10:00:00 | 1810.80 | 1805.03 | 1832.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 1816.00 | 1785.14 | 1815.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:45:00 | 1816.05 | 1785.14 | 1815.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 1818.95 | 1785.48 | 1815.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:30:00 | 1817.95 | 1785.48 | 1815.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 1800.30 | 1785.62 | 1815.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:45:00 | 1803.00 | 1785.62 | 1815.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 13:15:00 | 1810.45 | 1786.10 | 1815.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 13:30:00 | 1813.90 | 1786.10 | 1815.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 1813.40 | 1786.37 | 1815.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 14:45:00 | 1816.10 | 1786.37 | 1815.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 15:15:00 | 1814.00 | 1786.65 | 1815.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:15:00 | 1829.20 | 1786.65 | 1815.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 1801.80 | 1786.80 | 1815.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 10:45:00 | 1791.20 | 1786.83 | 1815.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 11:45:00 | 1800.85 | 1784.56 | 1811.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 12:15:00 | 1797.65 | 1784.56 | 1811.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 13:15:00 | 1836.95 | 1788.82 | 1811.21 | SL hit (close>static) qty=1.00 sl=1836.45 alert=retest2 |

### Cycle 4 — BUY (started 2025-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 14:15:00 | 1868.50 | 1825.61 | 1825.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 09:15:00 | 1885.00 | 1826.46 | 1825.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 13:15:00 | 1826.10 | 1830.30 | 1827.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 13:15:00 | 1826.10 | 1830.30 | 1827.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 13:15:00 | 1826.10 | 1830.30 | 1827.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 13:30:00 | 1831.90 | 1830.30 | 1827.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 14:15:00 | 1821.60 | 1830.21 | 1827.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 14:30:00 | 1819.25 | 1830.21 | 1827.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 15:15:00 | 1832.00 | 1830.23 | 1827.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 09:15:00 | 1842.00 | 1830.23 | 1827.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 10:30:00 | 1849.70 | 1830.43 | 1828.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 10:00:00 | 1845.10 | 1839.82 | 1833.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-10 11:15:00 | 1802.70 | 1839.41 | 1833.07 | SL hit (close<static) qty=1.00 sl=1818.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-01-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 14:15:00 | 1765.00 | 1827.20 | 1827.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 12:15:00 | 1748.40 | 1821.67 | 1824.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-21 09:15:00 | 1841.70 | 1817.29 | 1822.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 09:15:00 | 1841.70 | 1817.29 | 1822.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 1841.70 | 1817.29 | 1822.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 1841.70 | 1817.29 | 1822.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 1803.40 | 1817.15 | 1822.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 15:00:00 | 1799.85 | 1816.64 | 1821.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 09:15:00 | 1794.90 | 1814.86 | 1820.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 10:45:00 | 1798.35 | 1814.51 | 1820.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 11:30:00 | 1796.05 | 1814.48 | 1820.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 1709.86 | 1807.90 | 1816.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 1705.15 | 1807.90 | 1816.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 1708.43 | 1807.90 | 1816.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 1706.25 | 1807.90 | 1816.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-01 14:15:00 | 1789.90 | 1789.01 | 1804.72 | SL hit (close>ema200) qty=0.50 sl=1789.01 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 13:15:00 | 1668.90 | 1628.96 | 1628.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 11:15:00 | 1690.50 | 1631.26 | 1630.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 1680.80 | 1681.34 | 1661.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 13:00:00 | 1680.80 | 1681.34 | 1661.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1685.00 | 1712.81 | 1682.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:30:00 | 1684.70 | 1712.81 | 1682.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 1680.80 | 1712.49 | 1682.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 11:00:00 | 1680.80 | 1712.49 | 1682.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 1684.10 | 1712.20 | 1682.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 11:30:00 | 1680.30 | 1712.20 | 1682.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 1684.30 | 1711.93 | 1682.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 12:30:00 | 1684.10 | 1711.93 | 1682.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 1679.20 | 1711.60 | 1682.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:45:00 | 1678.20 | 1711.60 | 1682.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 1678.30 | 1711.27 | 1682.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 15:00:00 | 1678.30 | 1711.27 | 1682.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 1677.00 | 1710.93 | 1682.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:15:00 | 1662.60 | 1710.93 | 1682.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 1642.60 | 1709.57 | 1682.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 1642.60 | 1709.57 | 1682.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 1657.00 | 1666.81 | 1665.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:30:00 | 1651.90 | 1666.81 | 1665.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 1666.00 | 1666.79 | 1665.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:30:00 | 1665.90 | 1666.79 | 1665.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 1665.50 | 1666.78 | 1665.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 14:00:00 | 1665.50 | 1666.78 | 1665.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 14:15:00 | 1664.80 | 1666.76 | 1665.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 15:15:00 | 1660.00 | 1666.76 | 1665.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 1660.00 | 1666.69 | 1665.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:15:00 | 1657.20 | 1666.69 | 1665.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 1651.70 | 1666.43 | 1665.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:15:00 | 1646.00 | 1666.43 | 1665.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1668.20 | 1665.75 | 1665.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:30:00 | 1660.90 | 1665.75 | 1665.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 1671.40 | 1699.45 | 1685.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:45:00 | 1677.40 | 1699.45 | 1685.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 1666.10 | 1699.12 | 1685.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 1666.10 | 1699.12 | 1685.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 1683.00 | 1696.34 | 1684.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 13:30:00 | 1675.00 | 1696.34 | 1684.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 1689.20 | 1696.27 | 1684.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 10:15:00 | 1691.60 | 1696.12 | 1684.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 10:15:00 | 1674.10 | 1714.06 | 1704.05 | SL hit (close<static) qty=1.00 sl=1680.30 alert=retest2 |

### Cycle 7 — SELL (started 2025-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 12:15:00 | 1653.10 | 1698.92 | 1698.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 13:15:00 | 1650.00 | 1695.44 | 1697.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 10:15:00 | 1685.00 | 1684.72 | 1690.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-17 10:45:00 | 1684.80 | 1684.72 | 1690.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 1689.20 | 1684.76 | 1690.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:00:00 | 1689.20 | 1684.76 | 1690.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 1687.90 | 1684.79 | 1690.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:45:00 | 1696.40 | 1684.79 | 1690.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 1695.00 | 1684.90 | 1690.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:00:00 | 1695.00 | 1684.90 | 1690.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 1697.70 | 1685.02 | 1690.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:30:00 | 1698.70 | 1685.02 | 1690.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 1695.30 | 1685.12 | 1690.70 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 1803.20 | 1694.19 | 1694.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 15:15:00 | 1815.40 | 1700.73 | 1697.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 1735.10 | 1743.68 | 1723.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-21 10:00:00 | 1735.10 | 1743.68 | 1723.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 1726.60 | 1743.05 | 1723.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 14:45:00 | 1721.70 | 1743.05 | 1723.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1854.50 | 1853.66 | 1820.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 12:45:00 | 1873.00 | 1853.93 | 1821.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 13:30:00 | 1872.10 | 1854.19 | 1821.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 15:15:00 | 1871.60 | 1854.34 | 1821.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 10:45:00 | 1883.50 | 1854.91 | 1822.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-24 09:15:00 | 2060.30 | 1916.67 | 1872.62 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-10-31 09:15:00 | 1909.90 | 2024-11-04 14:15:00 | 1852.70 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2024-11-05 09:45:00 | 1878.95 | 2024-11-05 15:15:00 | 1858.00 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-11-05 10:45:00 | 1878.60 | 2024-11-05 15:15:00 | 1858.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-11-06 14:30:00 | 1876.70 | 2024-11-07 12:15:00 | 1859.95 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-12-05 10:45:00 | 1791.20 | 2024-12-12 13:15:00 | 1836.95 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2024-12-10 11:45:00 | 1800.85 | 2024-12-12 13:15:00 | 1836.95 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2024-12-10 12:15:00 | 1797.65 | 2024-12-12 13:15:00 | 1836.95 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2024-12-27 09:15:00 | 1796.45 | 2024-12-30 13:15:00 | 1839.00 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2025-01-07 09:15:00 | 1842.00 | 2025-01-10 11:15:00 | 1802.70 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-01-07 10:30:00 | 1849.70 | 2025-01-10 11:15:00 | 1802.70 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-01-10 10:00:00 | 1845.10 | 2025-01-10 11:15:00 | 1802.70 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-01-21 15:00:00 | 1799.85 | 2025-01-28 09:15:00 | 1709.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 09:15:00 | 1794.90 | 2025-01-28 09:15:00 | 1705.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 10:45:00 | 1798.35 | 2025-01-28 09:15:00 | 1708.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 11:30:00 | 1796.05 | 2025-01-28 09:15:00 | 1706.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 15:00:00 | 1799.85 | 2025-02-01 14:15:00 | 1789.90 | STOP_HIT | 0.50 | 0.55% |
| SELL | retest2 | 2025-01-23 09:15:00 | 1794.90 | 2025-02-01 14:15:00 | 1789.90 | STOP_HIT | 0.50 | 0.28% |
| SELL | retest2 | 2025-01-23 10:45:00 | 1798.35 | 2025-02-01 14:15:00 | 1789.90 | STOP_HIT | 0.50 | 0.47% |
| SELL | retest2 | 2025-01-23 11:30:00 | 1796.05 | 2025-02-01 14:15:00 | 1789.90 | STOP_HIT | 0.50 | 0.34% |
| SELL | retest2 | 2025-04-17 11:45:00 | 1613.80 | 2025-04-17 14:15:00 | 1649.60 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2025-04-23 10:00:00 | 1603.80 | 2025-05-06 12:15:00 | 1528.83 | PARTIAL | 0.50 | 4.67% |
| SELL | retest2 | 2025-04-30 13:45:00 | 1609.30 | 2025-05-06 13:15:00 | 1523.61 | PARTIAL | 0.50 | 5.32% |
| SELL | retest2 | 2025-04-23 10:00:00 | 1603.80 | 2025-05-13 14:15:00 | 1587.90 | STOP_HIT | 0.50 | 0.99% |
| SELL | retest2 | 2025-04-30 13:45:00 | 1609.30 | 2025-05-13 14:15:00 | 1587.90 | STOP_HIT | 0.50 | 1.33% |
| SELL | retest2 | 2025-05-15 09:45:00 | 1619.80 | 2025-05-16 12:15:00 | 1652.10 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-08-08 10:15:00 | 1691.60 | 2025-09-15 10:15:00 | 1674.10 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-09-17 12:30:00 | 1694.00 | 2025-09-25 12:15:00 | 1680.20 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-09-17 14:30:00 | 1691.60 | 2025-09-25 12:15:00 | 1680.20 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-09-18 12:45:00 | 1690.10 | 2025-09-25 12:15:00 | 1680.20 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-09-29 15:00:00 | 1707.00 | 2025-09-30 09:15:00 | 1687.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-09-30 14:45:00 | 1706.50 | 2025-10-01 12:15:00 | 1686.40 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-10-01 09:15:00 | 1713.60 | 2025-10-01 12:15:00 | 1686.40 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2026-01-30 12:45:00 | 1873.00 | 2026-02-24 09:15:00 | 2060.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-30 13:30:00 | 1872.10 | 2026-02-24 09:15:00 | 2059.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-30 15:15:00 | 1871.60 | 2026-02-24 09:15:00 | 2058.76 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-01 10:45:00 | 1883.50 | 2026-02-24 10:15:00 | 2071.85 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 11:15:00 | 1955.50 | 2026-05-06 10:15:00 | 2136.09 | TARGET_HIT | 1.00 | 9.23% |
| BUY | retest2 | 2026-04-06 12:00:00 | 1941.90 | 2026-05-06 11:15:00 | 2141.26 | TARGET_HIT | 1.00 | 10.27% |
| BUY | retest2 | 2026-04-07 10:00:00 | 1946.60 | 2026-05-06 13:15:00 | 2151.05 | TARGET_HIT | 1.00 | 10.50% |
