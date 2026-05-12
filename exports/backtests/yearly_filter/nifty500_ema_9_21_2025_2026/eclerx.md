# eClerx Services Ltd. (ECLERX)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1669.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 67 |
| ALERT1 | 47 |
| ALERT2 | 45 |
| ALERT2_SKIP | 24 |
| ALERT3 | 118 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 63 |
| PARTIAL | 15 |
| TARGET_HIT | 1 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 79 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 40 / 39
- **Target hits / Stop hits / Partials:** 1 / 63 / 15
- **Avg / median % per leg:** 1.10% / 0.04%
- **Sum % (uncompounded):** 87.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 6 | 28.6% | 0 | 21 | 0 | -1.12% | -23.5% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.26% | -1.3% |
| BUY @ 3rd Alert (retest2) | 20 | 6 | 30.0% | 0 | 20 | 0 | -1.11% | -22.2% |
| SELL (all) | 58 | 34 | 58.6% | 1 | 42 | 15 | 1.91% | 110.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 58 | 34 | 58.6% | 1 | 42 | 15 | 1.91% | 110.6% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.26% | -1.3% |
| retest2 (combined) | 78 | 40 | 51.3% | 1 | 62 | 15 | 1.13% | 88.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 1268.60 | 1227.32 | 1223.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 1279.40 | 1237.74 | 1228.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-21 15:15:00 | 1701.00 | 1702.23 | 1669.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-22 09:15:00 | 1694.50 | 1702.23 | 1669.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 1681.10 | 1689.46 | 1679.23 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 13:15:00 | 1668.25 | 1679.71 | 1679.83 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 09:15:00 | 1690.90 | 1681.20 | 1680.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 1736.40 | 1702.92 | 1692.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 12:15:00 | 1695.00 | 1714.82 | 1702.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 12:15:00 | 1695.00 | 1714.82 | 1702.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 1695.00 | 1714.82 | 1702.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:30:00 | 1703.55 | 1714.82 | 1702.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 1675.60 | 1706.98 | 1699.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:00:00 | 1675.60 | 1706.98 | 1699.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 1685.05 | 1702.59 | 1698.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:45:00 | 1674.15 | 1702.59 | 1698.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 1782.00 | 1765.22 | 1746.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 14:00:00 | 1787.00 | 1774.76 | 1767.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 09:15:00 | 1794.90 | 1778.58 | 1770.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 12:15:00 | 1795.55 | 1825.63 | 1825.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 1795.55 | 1825.63 | 1825.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 1732.70 | 1807.05 | 1817.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 14:15:00 | 1838.60 | 1813.36 | 1819.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 14:15:00 | 1838.60 | 1813.36 | 1819.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 1838.60 | 1813.36 | 1819.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:00:00 | 1838.60 | 1813.36 | 1819.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 1826.50 | 1815.99 | 1820.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 09:15:00 | 1788.20 | 1815.99 | 1820.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 10:15:00 | 1816.85 | 1800.11 | 1800.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 1816.85 | 1800.11 | 1800.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 12:15:00 | 1818.10 | 1806.00 | 1802.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 15:15:00 | 1801.50 | 1807.22 | 1804.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 15:15:00 | 1801.50 | 1807.22 | 1804.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 1801.50 | 1807.22 | 1804.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:15:00 | 1800.00 | 1807.22 | 1804.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 1799.00 | 1805.57 | 1803.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:30:00 | 1799.90 | 1805.57 | 1803.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 1795.05 | 1803.47 | 1803.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:30:00 | 1799.30 | 1803.47 | 1803.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 1793.85 | 1801.55 | 1802.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 12:15:00 | 1759.50 | 1793.14 | 1798.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 14:15:00 | 1789.55 | 1788.72 | 1795.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-18 15:00:00 | 1789.55 | 1788.72 | 1795.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 1824.50 | 1795.87 | 1797.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 09:15:00 | 1782.45 | 1795.87 | 1797.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 14:15:00 | 1693.33 | 1727.07 | 1749.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 1731.20 | 1725.57 | 1744.90 | SL hit (close>ema200) qty=0.50 sl=1725.57 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 1768.15 | 1750.59 | 1748.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 1787.90 | 1762.02 | 1755.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 12:15:00 | 1765.00 | 1769.71 | 1761.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 13:00:00 | 1765.00 | 1769.71 | 1761.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 1748.80 | 1765.52 | 1760.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:15:00 | 1749.00 | 1765.52 | 1760.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 1752.05 | 1762.83 | 1759.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 1766.10 | 1761.26 | 1759.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 10:15:00 | 1746.60 | 1756.14 | 1757.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 10:15:00 | 1746.60 | 1756.14 | 1757.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 11:15:00 | 1732.60 | 1750.23 | 1753.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 14:15:00 | 1754.15 | 1746.03 | 1750.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 14:15:00 | 1754.15 | 1746.03 | 1750.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 1754.15 | 1746.03 | 1750.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 14:30:00 | 1758.90 | 1746.03 | 1750.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 1761.55 | 1749.13 | 1751.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 1744.75 | 1749.13 | 1751.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1745.40 | 1748.39 | 1750.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 10:30:00 | 1736.80 | 1746.25 | 1749.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 09:45:00 | 1739.90 | 1739.63 | 1743.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 13:15:00 | 1732.80 | 1737.75 | 1741.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 13:00:00 | 1739.35 | 1728.80 | 1730.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 1715.00 | 1721.35 | 1726.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 15:15:00 | 1699.00 | 1715.73 | 1721.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 14:00:00 | 1701.45 | 1700.03 | 1709.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 15:15:00 | 1701.25 | 1701.84 | 1709.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:30:00 | 1700.65 | 1699.62 | 1706.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1678.45 | 1687.61 | 1696.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 11:15:00 | 1666.95 | 1687.01 | 1695.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 14:30:00 | 1677.65 | 1684.48 | 1690.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:15:00 | 1674.85 | 1684.09 | 1690.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:45:00 | 1672.90 | 1683.38 | 1689.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1694.30 | 1685.56 | 1689.75 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-10 14:15:00 | 1702.45 | 1692.89 | 1692.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 14:15:00 | 1702.45 | 1692.89 | 1692.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 09:15:00 | 1726.05 | 1701.46 | 1696.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 14:15:00 | 1777.35 | 1779.80 | 1761.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 15:00:00 | 1777.35 | 1779.80 | 1761.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1849.65 | 1849.93 | 1830.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:30:00 | 1824.30 | 1849.93 | 1830.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1860.35 | 1864.86 | 1855.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 1860.35 | 1864.86 | 1855.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 1843.85 | 1860.66 | 1854.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:00:00 | 1843.85 | 1860.66 | 1854.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 1848.55 | 1858.24 | 1853.82 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 14:15:00 | 1840.70 | 1850.65 | 1851.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 09:15:00 | 1826.80 | 1844.18 | 1847.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 09:15:00 | 1894.50 | 1843.02 | 1843.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 09:15:00 | 1894.50 | 1843.02 | 1843.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 1894.50 | 1843.02 | 1843.04 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-25 10:15:00 | 1850.50 | 1844.51 | 1843.72 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 15:15:00 | 1827.00 | 1842.49 | 1844.09 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 09:15:00 | 1893.00 | 1852.59 | 1848.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 11:15:00 | 1923.80 | 1889.92 | 1876.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 15:15:00 | 1885.00 | 1893.65 | 1883.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 09:15:00 | 1881.65 | 1893.65 | 1883.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1898.00 | 1894.52 | 1884.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 1914.75 | 1886.87 | 1884.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 1929.00 | 1905.18 | 1898.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 10:15:00 | 1991.95 | 2034.28 | 2038.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 10:15:00 | 1991.95 | 2034.28 | 2038.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 12:15:00 | 1968.30 | 1998.18 | 2008.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 09:15:00 | 1996.50 | 1979.56 | 1994.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 1996.50 | 1979.56 | 1994.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 1996.50 | 1979.56 | 1994.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:00:00 | 1996.50 | 1979.56 | 1994.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 2001.00 | 1983.85 | 1995.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:45:00 | 2003.60 | 1983.85 | 1995.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 2007.50 | 1988.58 | 1996.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:00:00 | 2007.50 | 1988.58 | 1996.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 15:15:00 | 2025.00 | 2004.21 | 2001.85 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-08-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 15:15:00 | 1993.20 | 2000.11 | 2000.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 1987.00 | 1995.79 | 1998.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 14:15:00 | 1998.20 | 1994.11 | 1997.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 14:15:00 | 1998.20 | 1994.11 | 1997.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 1998.20 | 1994.11 | 1997.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 1998.20 | 1994.11 | 1997.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 2002.00 | 1995.69 | 1997.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 1983.90 | 1995.69 | 1997.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1982.50 | 1993.05 | 1996.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 12:00:00 | 1971.40 | 1985.91 | 1992.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 09:15:00 | 2069.00 | 1986.62 | 1987.73 | SL hit (close>static) qty=1.00 sl=2016.45 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 10:15:00 | 2086.40 | 2006.58 | 1996.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 11:15:00 | 2101.50 | 2025.56 | 2006.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-28 09:15:00 | 2139.85 | 2155.01 | 2109.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 14:15:00 | 2119.35 | 2139.78 | 2118.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 2119.35 | 2139.78 | 2118.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 14:45:00 | 2111.15 | 2139.78 | 2118.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 2119.00 | 2135.62 | 2118.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:15:00 | 2094.45 | 2135.62 | 2118.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 2073.00 | 2123.10 | 2114.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 10:00:00 | 2073.00 | 2123.10 | 2114.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 2102.50 | 2121.43 | 2116.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 13:00:00 | 2102.50 | 2121.43 | 2116.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 2090.55 | 2115.25 | 2113.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 13:30:00 | 2083.45 | 2115.25 | 2113.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 15:15:00 | 2101.50 | 2112.49 | 2112.91 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 2162.50 | 2122.49 | 2117.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 10:15:00 | 2204.40 | 2138.88 | 2125.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 15:15:00 | 2244.95 | 2255.53 | 2218.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 09:15:00 | 2230.75 | 2255.53 | 2218.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 2238.10 | 2252.04 | 2220.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 10:30:00 | 2248.00 | 2244.49 | 2219.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 11:15:00 | 2194.35 | 2234.46 | 2217.58 | SL hit (close<static) qty=1.00 sl=2220.15 alert=retest2 |

### Cycle 20 — SELL (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 09:15:00 | 2184.95 | 2208.09 | 2209.22 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 13:15:00 | 2220.85 | 2211.33 | 2210.08 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 09:15:00 | 2191.70 | 2208.46 | 2209.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 2185.95 | 2201.00 | 2205.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 2180.90 | 2142.28 | 2159.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 2180.90 | 2142.28 | 2159.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 2180.90 | 2142.28 | 2159.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:45:00 | 2185.00 | 2142.28 | 2159.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 2187.50 | 2151.33 | 2162.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:00:00 | 2187.50 | 2151.33 | 2162.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 2177.50 | 2156.56 | 2163.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 13:30:00 | 2175.00 | 2165.00 | 2166.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 15:15:00 | 2180.00 | 2169.94 | 2168.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 15:15:00 | 2180.00 | 2169.94 | 2168.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 2222.00 | 2180.36 | 2173.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 14:15:00 | 2254.05 | 2255.15 | 2231.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 15:00:00 | 2254.05 | 2255.15 | 2231.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 2239.30 | 2249.88 | 2233.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:00:00 | 2239.30 | 2249.88 | 2233.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 2226.20 | 2245.14 | 2232.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:30:00 | 2223.50 | 2245.14 | 2232.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 2216.50 | 2239.41 | 2231.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:30:00 | 2221.90 | 2239.41 | 2231.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 15:15:00 | 2207.55 | 2225.94 | 2226.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 10:15:00 | 2188.80 | 2216.27 | 2222.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 2214.30 | 2197.59 | 2207.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 2214.30 | 2197.59 | 2207.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 2214.30 | 2197.59 | 2207.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 12:00:00 | 2183.25 | 2194.20 | 2204.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 12:30:00 | 2184.00 | 2191.73 | 2202.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 09:30:00 | 2179.45 | 2189.15 | 2197.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 12:00:00 | 2182.95 | 2188.35 | 2195.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 2195.85 | 2188.73 | 2193.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 15:00:00 | 2195.85 | 2188.73 | 2193.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 2200.00 | 2190.99 | 2194.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 2203.30 | 2190.99 | 2194.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 2224.40 | 2197.67 | 2197.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 09:15:00 | 2224.40 | 2197.67 | 2197.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 10:15:00 | 2254.20 | 2208.98 | 2202.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 12:15:00 | 2215.00 | 2215.14 | 2206.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 13:00:00 | 2215.00 | 2215.14 | 2206.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 2195.95 | 2211.30 | 2205.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:45:00 | 2194.80 | 2211.30 | 2205.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 2200.10 | 2209.06 | 2205.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:30:00 | 2191.15 | 2209.06 | 2205.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 2205.60 | 2208.37 | 2205.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:30:00 | 2235.95 | 2212.41 | 2207.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 10:15:00 | 2179.15 | 2205.76 | 2204.78 | SL hit (close<static) qty=1.00 sl=2196.55 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 11:15:00 | 2152.00 | 2195.01 | 2199.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 12:15:00 | 2132.60 | 2182.53 | 2193.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 12:15:00 | 2164.55 | 2156.23 | 2171.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-22 13:00:00 | 2164.55 | 2156.23 | 2171.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 2149.60 | 2147.76 | 2161.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:30:00 | 2147.55 | 2147.76 | 2161.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 2111.20 | 2073.50 | 2100.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:45:00 | 2112.40 | 2073.50 | 2100.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 2139.40 | 2086.68 | 2103.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:00:00 | 2139.40 | 2086.68 | 2103.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 2136.70 | 2096.68 | 2106.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:45:00 | 2145.15 | 2096.68 | 2106.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 2108.70 | 2105.04 | 2108.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:45:00 | 2141.25 | 2105.04 | 2108.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 2105.50 | 2105.13 | 2108.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 2064.05 | 2105.13 | 2108.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 13:00:00 | 2092.75 | 2105.76 | 2107.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 15:00:00 | 2069.10 | 2096.50 | 2103.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 10:15:00 | 1988.11 | 2008.51 | 2032.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 2003.85 | 1995.35 | 2013.44 | SL hit (close>ema200) qty=0.50 sl=1995.35 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 09:15:00 | 2040.60 | 2007.92 | 2003.54 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 11:15:00 | 1970.45 | 2006.42 | 2008.10 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 2027.90 | 2008.24 | 2006.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 12:15:00 | 2038.50 | 2014.29 | 2009.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 10:15:00 | 2054.70 | 2055.66 | 2040.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 12:15:00 | 2030.60 | 2050.44 | 2041.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 2030.60 | 2050.44 | 2041.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:00:00 | 2030.60 | 2050.44 | 2041.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 2020.95 | 2044.54 | 2039.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:45:00 | 2023.00 | 2044.54 | 2039.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 15:15:00 | 2015.25 | 2033.68 | 2034.88 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 2048.80 | 2036.70 | 2036.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 10:15:00 | 2074.50 | 2044.26 | 2039.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 13:15:00 | 2039.70 | 2047.27 | 2042.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 13:15:00 | 2039.70 | 2047.27 | 2042.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 2039.70 | 2047.27 | 2042.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:45:00 | 2037.35 | 2047.27 | 2042.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 2044.00 | 2046.62 | 2042.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 2057.20 | 2045.29 | 2042.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 11:00:00 | 2048.05 | 2047.30 | 2043.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 12:00:00 | 2052.00 | 2048.24 | 2044.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 13:15:00 | 2037.00 | 2051.54 | 2049.97 | SL hit (close<static) qty=1.00 sl=2039.70 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 14:15:00 | 2033.10 | 2047.85 | 2048.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 15:15:00 | 2032.60 | 2044.80 | 2047.00 | Break + close below crossover candle low |

### Cycle 33 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 2140.00 | 2063.84 | 2055.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 14:15:00 | 2174.45 | 2117.97 | 2088.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 2170.00 | 2171.91 | 2142.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 15:00:00 | 2170.00 | 2171.91 | 2142.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 2164.50 | 2171.76 | 2155.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:45:00 | 2156.05 | 2171.76 | 2155.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 2287.50 | 2340.64 | 2308.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:30:00 | 2266.65 | 2340.64 | 2308.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 2286.65 | 2329.84 | 2306.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:30:00 | 2283.35 | 2329.84 | 2306.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 2364.75 | 2366.87 | 2354.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 13:45:00 | 2353.35 | 2366.87 | 2354.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 2364.95 | 2365.99 | 2357.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 10:30:00 | 2404.45 | 2374.64 | 2361.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 11:30:00 | 2399.95 | 2378.70 | 2364.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 2336.60 | 2365.08 | 2363.24 | SL hit (close<static) qty=1.00 sl=2352.50 alert=retest2 |

### Cycle 34 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 10:15:00 | 2329.30 | 2357.92 | 2360.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 12:15:00 | 2315.50 | 2344.50 | 2353.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 14:15:00 | 2191.50 | 2167.61 | 2213.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 15:00:00 | 2191.50 | 2167.61 | 2213.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 2199.90 | 2173.25 | 2208.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 11:00:00 | 2171.30 | 2172.86 | 2204.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 12:45:00 | 2173.75 | 2171.47 | 2198.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 11:15:00 | 2240.00 | 2195.94 | 2198.93 | SL hit (close>static) qty=1.00 sl=2224.80 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 12:15:00 | 2257.05 | 2208.17 | 2204.21 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 2203.65 | 2218.05 | 2218.15 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 2245.10 | 2219.77 | 2217.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 12:15:00 | 2265.50 | 2232.54 | 2223.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 2244.60 | 2253.06 | 2238.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 2244.60 | 2253.06 | 2238.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 2244.60 | 2253.06 | 2238.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 2244.60 | 2253.06 | 2238.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 2237.95 | 2247.36 | 2239.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:45:00 | 2232.40 | 2247.36 | 2239.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 2233.95 | 2244.67 | 2238.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 13:45:00 | 2229.25 | 2244.67 | 2238.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 2225.75 | 2237.63 | 2236.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 09:15:00 | 2250.75 | 2237.63 | 2236.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 10:45:00 | 2249.25 | 2242.22 | 2238.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 11:15:00 | 2221.30 | 2238.04 | 2237.10 | SL hit (close<static) qty=1.00 sl=2222.50 alert=retest2 |

### Cycle 38 — SELL (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 12:15:00 | 2211.65 | 2232.76 | 2234.79 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 2248.45 | 2233.38 | 2232.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 14:15:00 | 2304.60 | 2247.63 | 2238.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 10:15:00 | 2243.55 | 2255.67 | 2245.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 10:15:00 | 2243.55 | 2255.67 | 2245.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 2243.55 | 2255.67 | 2245.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:45:00 | 2247.25 | 2255.67 | 2245.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 2247.50 | 2254.04 | 2245.84 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 15:15:00 | 2227.50 | 2241.58 | 2242.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 09:15:00 | 2219.35 | 2235.87 | 2238.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 14:15:00 | 2214.15 | 2212.04 | 2219.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 15:00:00 | 2214.15 | 2212.04 | 2219.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 2229.05 | 2215.76 | 2219.91 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 12:15:00 | 2241.25 | 2226.09 | 2224.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 13:15:00 | 2247.30 | 2230.33 | 2226.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 15:15:00 | 2231.25 | 2233.42 | 2228.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 09:15:00 | 2248.30 | 2233.42 | 2228.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 2285.95 | 2243.93 | 2233.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 10:30:00 | 2296.40 | 2251.88 | 2238.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 2322.75 | 2266.83 | 2251.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 2353.90 | 2404.15 | 2405.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 2353.90 | 2404.15 | 2405.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 2347.60 | 2392.84 | 2400.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 14:15:00 | 2384.55 | 2374.64 | 2387.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 14:15:00 | 2384.55 | 2374.64 | 2387.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 2384.55 | 2374.64 | 2387.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:45:00 | 2396.50 | 2374.64 | 2387.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 15:15:00 | 2373.40 | 2374.39 | 2386.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 09:15:00 | 2339.90 | 2374.39 | 2386.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 09:15:00 | 2222.91 | 2286.93 | 2326.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 12:15:00 | 2219.50 | 2212.02 | 2252.80 | SL hit (close>ema200) qty=0.50 sl=2212.02 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 10:15:00 | 2285.90 | 2253.50 | 2252.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 11:15:00 | 2295.10 | 2261.82 | 2256.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 2279.50 | 2292.45 | 2275.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:00:00 | 2279.50 | 2292.45 | 2275.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 2276.85 | 2289.33 | 2276.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:30:00 | 2276.75 | 2289.33 | 2276.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 2267.40 | 2284.94 | 2275.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:30:00 | 2274.75 | 2284.94 | 2275.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 2257.05 | 2279.37 | 2273.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 2257.05 | 2279.37 | 2273.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 2242.00 | 2271.89 | 2270.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:00:00 | 2242.00 | 2271.89 | 2270.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 2230.50 | 2263.61 | 2267.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 15:15:00 | 2220.25 | 2240.55 | 2252.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 2244.00 | 2236.21 | 2242.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 2244.00 | 2236.21 | 2242.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 2244.00 | 2236.21 | 2242.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 2253.00 | 2236.21 | 2242.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 2254.05 | 2239.78 | 2243.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:30:00 | 2260.20 | 2239.78 | 2243.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 2242.50 | 2240.32 | 2243.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:30:00 | 2249.45 | 2240.32 | 2243.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 2260.20 | 2243.86 | 2244.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:30:00 | 2259.20 | 2243.86 | 2244.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 2262.00 | 2247.49 | 2246.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 15:15:00 | 2275.00 | 2252.99 | 2248.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 2305.35 | 2306.26 | 2284.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:45:00 | 2307.65 | 2306.26 | 2284.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 2365.60 | 2380.49 | 2356.44 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 2325.10 | 2349.73 | 2350.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 14:15:00 | 2281.10 | 2324.39 | 2336.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 2294.25 | 2273.87 | 2294.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 10:15:00 | 2294.25 | 2273.87 | 2294.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 2294.25 | 2273.87 | 2294.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 2294.25 | 2273.87 | 2294.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 2296.70 | 2278.44 | 2295.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:30:00 | 2297.50 | 2278.44 | 2295.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 2304.00 | 2283.55 | 2295.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:00:00 | 2304.00 | 2283.55 | 2295.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 2303.40 | 2287.52 | 2296.55 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 15:15:00 | 2349.50 | 2308.95 | 2305.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 11:15:00 | 2396.35 | 2336.76 | 2319.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 10:15:00 | 2373.05 | 2379.85 | 2353.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 10:45:00 | 2372.30 | 2379.85 | 2353.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 2361.10 | 2372.00 | 2356.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 10:15:00 | 2378.35 | 2367.42 | 2357.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 09:15:00 | 2370.90 | 2391.35 | 2393.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 2370.90 | 2391.35 | 2393.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 2357.85 | 2375.11 | 2382.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 2308.00 | 2299.41 | 2322.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 2302.50 | 2299.41 | 2322.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 2301.00 | 2299.73 | 2320.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 2278.25 | 2294.23 | 2311.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 2336.95 | 2306.51 | 2308.47 | SL hit (close>static) qty=1.00 sl=2325.00 alert=retest2 |

### Cycle 49 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 2343.95 | 2313.99 | 2311.69 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 2267.00 | 2309.89 | 2312.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 12:15:00 | 2238.50 | 2281.38 | 2297.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 2220.70 | 2139.37 | 2172.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 2220.70 | 2139.37 | 2172.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 2220.70 | 2139.37 | 2172.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 2237.00 | 2139.37 | 2172.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 2221.30 | 2155.75 | 2177.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:45:00 | 2232.00 | 2155.75 | 2177.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 2179.05 | 2167.33 | 2179.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 13:15:00 | 2165.45 | 2167.33 | 2179.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 15:00:00 | 2175.40 | 2169.37 | 2177.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:15:00 | 2177.45 | 2173.88 | 2178.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 15:15:00 | 2230.00 | 2175.12 | 2170.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 2230.00 | 2175.12 | 2170.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 09:15:00 | 2285.65 | 2219.43 | 2198.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 2295.00 | 2302.24 | 2261.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 10:00:00 | 2295.00 | 2302.24 | 2261.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 2371.50 | 2334.71 | 2313.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 10:15:00 | 2384.40 | 2334.71 | 2313.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 13:15:00 | 2388.70 | 2350.49 | 2326.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 10:15:00 | 2247.70 | 2383.52 | 2379.15 | SL hit (close<static) qty=1.00 sl=2282.95 alert=retest2 |

### Cycle 52 — SELL (started 2026-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 11:15:00 | 2247.50 | 2356.32 | 2367.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 2173.60 | 2238.87 | 2278.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 2147.35 | 2143.76 | 2201.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 10:00:00 | 2147.35 | 2143.76 | 2201.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 1886.10 | 1862.29 | 1914.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:30:00 | 1888.15 | 1862.29 | 1914.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 1901.75 | 1846.63 | 1866.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 1901.75 | 1846.63 | 1866.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 1834.05 | 1844.11 | 1863.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 12:45:00 | 1816.30 | 1838.77 | 1857.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:30:00 | 1817.80 | 1814.45 | 1825.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 10:45:00 | 1825.30 | 1816.65 | 1825.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 13:15:00 | 1734.03 | 1762.90 | 1781.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:15:00 | 1725.48 | 1742.81 | 1766.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:15:00 | 1726.91 | 1742.81 | 1766.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-25 12:15:00 | 1642.77 | 1676.36 | 1709.10 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 53 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 1572.90 | 1538.44 | 1536.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 1609.75 | 1565.12 | 1550.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 1559.85 | 1571.83 | 1560.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 1559.85 | 1571.83 | 1560.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 1559.85 | 1571.83 | 1560.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 1559.85 | 1571.83 | 1560.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 1559.00 | 1569.26 | 1560.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 1544.80 | 1569.26 | 1560.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1529.50 | 1561.31 | 1557.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 1522.20 | 1561.31 | 1557.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 1579.70 | 1564.99 | 1559.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 13:00:00 | 1594.15 | 1572.83 | 1564.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 1504.00 | 1562.50 | 1562.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 1504.00 | 1562.50 | 1562.77 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 15:15:00 | 1604.00 | 1556.14 | 1555.72 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-03-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 09:15:00 | 1512.30 | 1547.37 | 1551.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 1504.70 | 1538.83 | 1547.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 1533.10 | 1530.06 | 1539.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 1533.10 | 1530.06 | 1539.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1533.10 | 1530.06 | 1539.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 1534.00 | 1530.06 | 1539.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1522.00 | 1527.96 | 1537.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:15:00 | 1488.20 | 1516.22 | 1529.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 13:15:00 | 1487.60 | 1511.24 | 1526.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 14:45:00 | 1483.10 | 1499.46 | 1518.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 12:00:00 | 1489.00 | 1496.00 | 1510.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 1469.10 | 1478.69 | 1491.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:45:00 | 1480.80 | 1478.69 | 1491.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1479.10 | 1473.59 | 1485.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:30:00 | 1460.90 | 1472.33 | 1480.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 13:15:00 | 1413.79 | 1444.90 | 1462.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 13:15:00 | 1413.22 | 1444.90 | 1462.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 13:15:00 | 1414.55 | 1444.90 | 1462.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 14:15:00 | 1408.94 | 1436.34 | 1456.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 1425.30 | 1422.58 | 1442.63 | SL hit (close>ema200) qty=0.50 sl=1422.58 alert=retest2 |

### Cycle 57 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1482.60 | 1454.11 | 1452.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 1498.00 | 1462.88 | 1456.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1432.40 | 1475.63 | 1468.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1432.40 | 1475.63 | 1468.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1432.40 | 1475.63 | 1468.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 1432.40 | 1475.63 | 1468.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 1428.00 | 1466.10 | 1465.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:30:00 | 1427.60 | 1466.10 | 1465.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 1473.80 | 1467.74 | 1466.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:30:00 | 1469.30 | 1467.74 | 1466.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 1448.50 | 1463.89 | 1464.44 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 14:15:00 | 1470.00 | 1465.11 | 1464.95 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 1407.00 | 1454.27 | 1460.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 1397.50 | 1442.92 | 1454.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1446.00 | 1416.79 | 1432.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1446.00 | 1416.79 | 1432.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1446.00 | 1416.79 | 1432.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 1446.00 | 1416.79 | 1432.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1414.30 | 1416.29 | 1430.99 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 1486.10 | 1441.31 | 1438.74 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 1418.80 | 1436.97 | 1437.77 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 1449.10 | 1437.99 | 1437.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 09:15:00 | 1458.30 | 1445.24 | 1441.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 15:15:00 | 1493.00 | 1496.16 | 1482.05 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:15:00 | 1513.70 | 1496.16 | 1482.05 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 1494.60 | 1527.93 | 1512.18 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-09 09:15:00 | 1494.60 | 1527.93 | 1512.18 | SL hit (close<ema400) qty=1.00 sl=1512.18 alert=retest1 |

### Cycle 64 — SELL (started 2026-04-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 15:15:00 | 1492.00 | 1505.28 | 1505.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 09:15:00 | 1471.80 | 1493.23 | 1498.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 1498.80 | 1472.20 | 1482.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 1498.80 | 1472.20 | 1482.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1498.80 | 1472.20 | 1482.33 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2026-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 09:15:00 | 1555.10 | 1498.33 | 1491.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 1606.50 | 1555.71 | 1529.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 1598.30 | 1609.74 | 1576.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 09:30:00 | 1590.00 | 1609.74 | 1576.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 1602.00 | 1618.23 | 1598.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 09:45:00 | 1603.50 | 1618.23 | 1598.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 1596.80 | 1613.94 | 1598.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:45:00 | 1597.10 | 1613.94 | 1598.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 1583.60 | 1607.87 | 1597.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 12:00:00 | 1583.60 | 1607.87 | 1597.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-04-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 14:15:00 | 1565.00 | 1589.55 | 1590.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 1524.60 | 1573.11 | 1582.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1496.80 | 1474.20 | 1490.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1496.80 | 1474.20 | 1490.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1496.80 | 1474.20 | 1490.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1505.00 | 1474.20 | 1490.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1485.10 | 1476.38 | 1490.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 12:30:00 | 1484.60 | 1480.37 | 1489.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:15:00 | 1483.00 | 1480.37 | 1489.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:45:00 | 1482.90 | 1481.24 | 1489.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 15:15:00 | 1410.37 | 1435.62 | 1452.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 15:15:00 | 1408.85 | 1435.62 | 1452.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 15:15:00 | 1408.76 | 1435.62 | 1452.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-30 11:15:00 | 1442.30 | 1428.00 | 1443.59 | SL hit (close>ema200) qty=0.50 sl=1428.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 1447.70 | 1432.97 | 1431.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 1499.20 | 1452.17 | 1441.47 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-05 14:00:00 | 1787.00 | 2025-06-12 12:15:00 | 1795.55 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2025-06-06 09:15:00 | 1794.90 | 2025-06-12 12:15:00 | 1795.55 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-06-13 09:15:00 | 1788.20 | 2025-06-17 10:15:00 | 1816.85 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-06-19 09:15:00 | 1782.45 | 2025-06-20 14:15:00 | 1693.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-19 09:15:00 | 1782.45 | 2025-06-23 09:15:00 | 1731.20 | STOP_HIT | 0.50 | 2.88% |
| BUY | retest2 | 2025-06-26 09:15:00 | 1766.10 | 2025-06-26 10:15:00 | 1746.60 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-06-30 10:30:00 | 1736.80 | 2025-07-10 14:15:00 | 1702.45 | STOP_HIT | 1.00 | 1.98% |
| SELL | retest2 | 2025-07-01 09:45:00 | 1739.90 | 2025-07-10 14:15:00 | 1702.45 | STOP_HIT | 1.00 | 2.15% |
| SELL | retest2 | 2025-07-01 13:15:00 | 1732.80 | 2025-07-10 14:15:00 | 1702.45 | STOP_HIT | 1.00 | 1.75% |
| SELL | retest2 | 2025-07-03 13:00:00 | 1739.35 | 2025-07-10 14:15:00 | 1702.45 | STOP_HIT | 1.00 | 2.12% |
| SELL | retest2 | 2025-07-04 15:15:00 | 1699.00 | 2025-07-10 14:15:00 | 1702.45 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-07-07 14:00:00 | 1701.45 | 2025-07-10 14:15:00 | 1702.45 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-07-07 15:15:00 | 1701.25 | 2025-07-10 14:15:00 | 1702.45 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-07-08 09:30:00 | 1700.65 | 2025-07-10 14:15:00 | 1702.45 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-07-09 11:15:00 | 1666.95 | 2025-07-10 14:15:00 | 1702.45 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-07-09 14:30:00 | 1677.65 | 2025-07-10 14:15:00 | 1702.45 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-07-10 09:15:00 | 1674.85 | 2025-07-10 14:15:00 | 1702.45 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-07-10 09:45:00 | 1672.90 | 2025-07-10 14:15:00 | 1702.45 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-08-01 09:15:00 | 1914.75 | 2025-08-13 10:15:00 | 1991.95 | STOP_HIT | 1.00 | 4.03% |
| BUY | retest2 | 2025-08-04 09:15:00 | 1929.00 | 2025-08-13 10:15:00 | 1991.95 | STOP_HIT | 1.00 | 3.26% |
| SELL | retest2 | 2025-08-22 12:00:00 | 1971.40 | 2025-08-25 09:15:00 | 2069.00 | STOP_HIT | 1.00 | -4.95% |
| BUY | retest2 | 2025-09-03 10:30:00 | 2248.00 | 2025-09-03 11:15:00 | 2194.35 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2025-09-09 13:30:00 | 2175.00 | 2025-09-09 15:15:00 | 2180.00 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-09-16 12:00:00 | 2183.25 | 2025-09-18 09:15:00 | 2224.40 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-09-16 12:30:00 | 2184.00 | 2025-09-18 09:15:00 | 2224.40 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-09-17 09:30:00 | 2179.45 | 2025-09-18 09:15:00 | 2224.40 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-09-17 12:00:00 | 2182.95 | 2025-09-18 09:15:00 | 2224.40 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-09-19 09:30:00 | 2235.95 | 2025-09-19 10:15:00 | 2179.15 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2025-09-26 09:15:00 | 2064.05 | 2025-10-01 10:15:00 | 1988.11 | PARTIAL | 0.50 | 3.68% |
| SELL | retest2 | 2025-09-26 09:15:00 | 2064.05 | 2025-10-03 09:15:00 | 2003.85 | STOP_HIT | 0.50 | 2.92% |
| SELL | retest2 | 2025-09-26 13:00:00 | 2092.75 | 2025-10-03 13:15:00 | 1960.85 | PARTIAL | 0.50 | 6.30% |
| SELL | retest2 | 2025-09-26 15:00:00 | 2069.10 | 2025-10-03 13:15:00 | 1965.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-26 13:00:00 | 2092.75 | 2025-10-06 09:15:00 | 2003.95 | STOP_HIT | 0.50 | 4.24% |
| SELL | retest2 | 2025-09-26 15:00:00 | 2069.10 | 2025-10-06 09:15:00 | 2003.95 | STOP_HIT | 0.50 | 3.15% |
| BUY | retest2 | 2025-10-16 09:15:00 | 2057.20 | 2025-10-17 13:15:00 | 2037.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-10-16 11:00:00 | 2048.05 | 2025-10-17 13:15:00 | 2037.00 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-10-16 12:00:00 | 2052.00 | 2025-10-17 13:15:00 | 2037.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-11-04 10:30:00 | 2404.45 | 2025-11-06 09:15:00 | 2336.60 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2025-11-04 11:30:00 | 2399.95 | 2025-11-06 09:15:00 | 2336.60 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2025-11-11 11:00:00 | 2171.30 | 2025-11-12 11:15:00 | 2240.00 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2025-11-11 12:45:00 | 2173.75 | 2025-11-12 11:15:00 | 2240.00 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2025-11-19 09:15:00 | 2250.75 | 2025-11-19 11:15:00 | 2221.30 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-11-19 10:45:00 | 2249.25 | 2025-11-19 11:15:00 | 2221.30 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-11-28 10:30:00 | 2296.40 | 2025-12-08 09:15:00 | 2353.90 | STOP_HIT | 1.00 | 2.50% |
| BUY | retest2 | 2025-12-01 09:15:00 | 2322.75 | 2025-12-08 09:15:00 | 2353.90 | STOP_HIT | 1.00 | 1.34% |
| SELL | retest2 | 2025-12-09 09:15:00 | 2339.90 | 2025-12-10 09:15:00 | 2222.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-09 09:15:00 | 2339.90 | 2025-12-11 12:15:00 | 2219.50 | STOP_HIT | 0.50 | 5.15% |
| BUY | retest2 | 2026-01-05 10:15:00 | 2378.35 | 2026-01-08 09:15:00 | 2370.90 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2026-01-13 14:00:00 | 2278.25 | 2026-01-16 09:15:00 | 2336.95 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2026-01-22 13:15:00 | 2165.45 | 2026-01-27 15:15:00 | 2230.00 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2026-01-22 15:00:00 | 2175.40 | 2026-01-27 15:15:00 | 2230.00 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2026-01-23 11:15:00 | 2177.45 | 2026-01-27 15:15:00 | 2230.00 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2026-02-02 10:15:00 | 2384.40 | 2026-02-04 10:15:00 | 2247.70 | STOP_HIT | 1.00 | -5.73% |
| BUY | retest2 | 2026-02-02 13:15:00 | 2388.70 | 2026-02-04 10:15:00 | 2247.70 | STOP_HIT | 1.00 | -5.90% |
| SELL | retest2 | 2026-02-17 12:45:00 | 1816.30 | 2026-02-23 13:15:00 | 1734.03 | PARTIAL | 0.50 | 4.53% |
| SELL | retest2 | 2026-02-19 09:30:00 | 1817.80 | 2026-02-24 09:15:00 | 1725.48 | PARTIAL | 0.50 | 5.08% |
| SELL | retest2 | 2026-02-19 10:45:00 | 1825.30 | 2026-02-24 09:15:00 | 1726.91 | PARTIAL | 0.50 | 5.39% |
| SELL | retest2 | 2026-02-17 12:45:00 | 1816.30 | 2026-02-25 12:15:00 | 1642.77 | TARGET_HIT | 0.50 | 9.55% |
| SELL | retest2 | 2026-02-19 09:30:00 | 1817.80 | 2026-02-26 09:15:00 | 1662.85 | STOP_HIT | 0.50 | 8.52% |
| SELL | retest2 | 2026-02-19 10:45:00 | 1825.30 | 2026-02-26 09:15:00 | 1662.85 | STOP_HIT | 0.50 | 8.90% |
| BUY | retest2 | 2026-03-12 13:00:00 | 1594.15 | 2026-03-13 09:15:00 | 1504.00 | STOP_HIT | 1.00 | -5.66% |
| SELL | retest2 | 2026-03-17 12:15:00 | 1488.20 | 2026-03-23 13:15:00 | 1413.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-17 13:15:00 | 1487.60 | 2026-03-23 13:15:00 | 1413.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-17 14:45:00 | 1483.10 | 2026-03-23 13:15:00 | 1414.55 | PARTIAL | 0.50 | 4.62% |
| SELL | retest2 | 2026-03-18 12:00:00 | 1489.00 | 2026-03-23 14:15:00 | 1408.94 | PARTIAL | 0.50 | 5.38% |
| SELL | retest2 | 2026-03-17 12:15:00 | 1488.20 | 2026-03-24 11:15:00 | 1425.30 | STOP_HIT | 0.50 | 4.23% |
| SELL | retest2 | 2026-03-17 13:15:00 | 1487.60 | 2026-03-24 11:15:00 | 1425.30 | STOP_HIT | 0.50 | 4.19% |
| SELL | retest2 | 2026-03-17 14:45:00 | 1483.10 | 2026-03-24 11:15:00 | 1425.30 | STOP_HIT | 0.50 | 3.90% |
| SELL | retest2 | 2026-03-18 12:00:00 | 1489.00 | 2026-03-24 11:15:00 | 1425.30 | STOP_HIT | 0.50 | 4.28% |
| SELL | retest2 | 2026-03-20 14:30:00 | 1460.90 | 2026-03-25 09:15:00 | 1482.60 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2026-03-24 13:30:00 | 1464.90 | 2026-03-25 09:15:00 | 1482.60 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-03-24 14:15:00 | 1464.40 | 2026-03-25 09:15:00 | 1482.60 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest1 | 2026-04-08 09:15:00 | 1513.70 | 2026-04-09 09:15:00 | 1494.60 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-04-27 12:30:00 | 1484.60 | 2026-04-29 15:15:00 | 1410.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-27 13:15:00 | 1483.00 | 2026-04-29 15:15:00 | 1408.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-27 13:45:00 | 1482.90 | 2026-04-29 15:15:00 | 1408.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-27 12:30:00 | 1484.60 | 2026-04-30 11:15:00 | 1442.30 | STOP_HIT | 0.50 | 2.85% |
| SELL | retest2 | 2026-04-27 13:15:00 | 1483.00 | 2026-04-30 11:15:00 | 1442.30 | STOP_HIT | 0.50 | 2.74% |
| SELL | retest2 | 2026-04-27 13:45:00 | 1482.90 | 2026-04-30 11:15:00 | 1442.30 | STOP_HIT | 0.50 | 2.74% |
