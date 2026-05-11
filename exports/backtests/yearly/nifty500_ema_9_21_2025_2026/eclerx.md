# eClerx Services Ltd. (ECLERX)

## Backtest Summary

- **Window:** 2024-08-22 09:15:00 → 2026-05-08 15:30:00 (2939 bars)
- **Last close:** 1668.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 69 |
| ALERT1 | 46 |
| ALERT2 | 45 |
| ALERT2_SKIP | 24 |
| ALERT3 | 319 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 0 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 0 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 0
- **Target hits / Stop hits / Partials:** 0 / 0 / 0
- **Avg / median % per leg:** 0.00% / 0.00%
- **Sum % (uncompounded):** 0.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 1268.60 | 1228.48 | 1224.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 1279.40 | 1238.67 | 1229.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-21 15:15:00 | 1701.00 | 1702.10 | 1669.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-21 15:15:00 | 1701.00 | 1702.10 | 1669.07 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 1681.10 | 1689.45 | 1679.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:15:00 | 1681.10 | 1689.45 | 1679.19 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 1679.75 | 1687.44 | 1680.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:15:00 | 1679.75 | 1687.44 | 1680.05 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 1690.00 | 1687.95 | 1680.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:15:00 | 1690.00 | 1687.95 | 1680.95 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 1680.15 | 1686.08 | 1681.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 14:15:00 | 1680.15 | 1686.08 | 1681.30 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 1672.55 | 1683.37 | 1680.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 15:15:00 | 1672.55 | 1683.37 | 1680.50 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 1683.50 | 1682.90 | 1680.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 10:15:00 | 1683.50 | 1682.90 | 1680.75 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 1681.55 | 1682.63 | 1680.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:15:00 | 1681.55 | 1682.63 | 1680.83 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 1682.35 | 1682.57 | 1680.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:15:00 | 1682.35 | 1682.57 | 1680.97 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 13:15:00 | 1668.25 | 1679.71 | 1679.81 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 09:15:00 | 1690.00 | 1681.02 | 1680.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 11:15:00 | 1702.25 | 1687.49 | 1683.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 12:15:00 | 1695.00 | 1714.96 | 1702.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 12:15:00 | 1695.00 | 1714.96 | 1702.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 1695.00 | 1714.96 | 1702.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:15:00 | 1695.00 | 1714.96 | 1702.19 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 1675.60 | 1707.09 | 1699.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 13:15:00 | 1675.60 | 1707.09 | 1699.77 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 1685.05 | 1702.68 | 1698.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:15:00 | 1685.05 | 1702.68 | 1698.43 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 1782.00 | 1767.08 | 1747.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 1782.00 | 1767.08 | 1747.82 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 1733.45 | 1762.28 | 1753.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:15:00 | 1733.45 | 1762.28 | 1753.51 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 1737.45 | 1757.32 | 1752.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:15:00 | 1737.45 | 1757.32 | 1752.05 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 11:15:00 | 1755.45 | 1755.71 | 1752.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:15:00 | 1755.45 | 1755.71 | 1752.49 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 1750.45 | 1754.66 | 1752.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 12:15:00 | 1750.45 | 1754.66 | 1752.30 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 1757.00 | 1755.13 | 1752.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:15:00 | 1757.00 | 1755.13 | 1752.73 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 1762.00 | 1756.50 | 1753.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 14:15:00 | 1762.00 | 1756.50 | 1753.57 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 1742.50 | 1753.70 | 1752.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 15:15:00 | 1742.50 | 1753.70 | 1752.57 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 1756.35 | 1765.27 | 1760.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:15:00 | 1756.35 | 1765.27 | 1760.25 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 1754.25 | 1763.07 | 1759.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:15:00 | 1754.25 | 1763.07 | 1759.71 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 1822.35 | 1813.08 | 1799.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 14:15:00 | 1822.35 | 1813.08 | 1799.95 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 1825.05 | 1841.68 | 1827.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:15:00 | 1825.05 | 1841.68 | 1827.18 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 1838.70 | 1841.08 | 1828.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:15:00 | 1838.70 | 1841.08 | 1828.22 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 1825.00 | 1837.87 | 1827.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:15:00 | 1825.00 | 1837.87 | 1827.93 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 1845.00 | 1839.29 | 1829.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:15:00 | 1845.00 | 1839.29 | 1829.48 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 1838.05 | 1839.05 | 1830.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:15:00 | 1838.05 | 1839.05 | 1830.26 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 1825.00 | 1837.55 | 1831.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:15:00 | 1825.00 | 1837.55 | 1831.19 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 1825.00 | 1835.04 | 1830.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:15:00 | 1825.00 | 1835.04 | 1830.62 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 1795.55 | 1827.14 | 1827.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 1732.70 | 1808.25 | 1818.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 14:15:00 | 1838.05 | 1814.21 | 1820.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 14:15:00 | 1838.05 | 1814.21 | 1820.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 1838.05 | 1814.21 | 1820.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:15:00 | 1838.05 | 1814.21 | 1820.57 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 1833.75 | 1818.12 | 1821.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:15:00 | 1833.75 | 1818.12 | 1821.77 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 1800.00 | 1787.19 | 1796.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:15:00 | 1800.00 | 1787.19 | 1796.57 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1804.00 | 1790.55 | 1797.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:15:00 | 1804.00 | 1790.55 | 1797.24 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 1816.85 | 1800.13 | 1800.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:15:00 | 1816.85 | 1800.13 | 1800.40 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 11:15:00 | 1814.45 | 1803.00 | 1801.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 12:15:00 | 1818.10 | 1806.02 | 1803.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 15:15:00 | 1801.50 | 1807.23 | 1804.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 15:15:00 | 1801.50 | 1807.23 | 1804.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 1801.50 | 1807.23 | 1804.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 15:15:00 | 1801.50 | 1807.23 | 1804.63 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 1799.00 | 1805.58 | 1804.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:15:00 | 1799.00 | 1805.58 | 1804.11 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 1795.05 | 1803.47 | 1803.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:15:00 | 1795.05 | 1803.47 | 1803.29 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 1793.85 | 1801.55 | 1802.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 12:15:00 | 1759.50 | 1793.14 | 1798.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 14:15:00 | 1789.55 | 1788.72 | 1795.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-18 14:15:00 | 1789.55 | 1788.72 | 1795.36 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 1808.40 | 1792.66 | 1796.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 15:15:00 | 1808.40 | 1792.66 | 1796.54 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1774.70 | 1789.06 | 1794.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 1774.70 | 1789.06 | 1794.56 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 1745.25 | 1734.00 | 1744.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:15:00 | 1745.25 | 1734.00 | 1744.13 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 1749.50 | 1737.10 | 1744.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 13:15:00 | 1749.50 | 1737.10 | 1744.62 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 1744.50 | 1738.58 | 1744.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:15:00 | 1744.50 | 1738.58 | 1744.61 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 1740.85 | 1739.04 | 1744.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 15:15:00 | 1740.85 | 1739.04 | 1744.26 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 1764.15 | 1744.06 | 1746.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:15:00 | 1764.15 | 1744.06 | 1746.07 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 1762.10 | 1747.67 | 1747.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 12:15:00 | 1774.25 | 1756.26 | 1751.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 1755.65 | 1757.54 | 1753.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 14:15:00 | 1755.65 | 1757.54 | 1753.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 1755.65 | 1757.54 | 1753.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 14:15:00 | 1755.65 | 1757.54 | 1753.13 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 1750.00 | 1756.03 | 1752.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:15:00 | 1750.00 | 1756.03 | 1752.84 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 1748.80 | 1765.68 | 1760.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 13:15:00 | 1748.80 | 1765.68 | 1760.53 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 1752.75 | 1763.10 | 1759.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:15:00 | 1752.75 | 1763.10 | 1759.83 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 1751.55 | 1760.79 | 1759.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 15:15:00 | 1751.55 | 1760.79 | 1759.07 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 10:15:00 | 1746.60 | 1755.84 | 1756.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 11:15:00 | 1732.70 | 1750.20 | 1753.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 14:15:00 | 1754.40 | 1746.07 | 1750.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 14:15:00 | 1754.40 | 1746.07 | 1750.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 1754.40 | 1746.07 | 1750.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 14:15:00 | 1754.40 | 1746.07 | 1750.10 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 1754.80 | 1747.81 | 1750.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:15:00 | 1754.80 | 1747.81 | 1750.53 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1745.15 | 1747.28 | 1750.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 1745.15 | 1747.28 | 1750.04 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 1752.15 | 1739.64 | 1744.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 14:15:00 | 1752.15 | 1739.64 | 1744.38 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 1746.60 | 1741.04 | 1744.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 15:15:00 | 1746.60 | 1741.04 | 1744.58 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 1740.25 | 1737.70 | 1741.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:15:00 | 1740.25 | 1737.70 | 1741.51 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 1739.80 | 1736.83 | 1740.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 14:15:00 | 1739.80 | 1736.83 | 1740.37 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 1736.30 | 1736.73 | 1740.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 15:15:00 | 1736.30 | 1736.73 | 1740.00 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 1718.90 | 1733.16 | 1738.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 1718.90 | 1733.16 | 1738.08 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1715.95 | 1722.17 | 1729.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:15:00 | 1715.95 | 1722.17 | 1729.09 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 1733.70 | 1724.47 | 1729.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:15:00 | 1733.70 | 1724.47 | 1729.51 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 1733.95 | 1726.37 | 1729.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:15:00 | 1733.95 | 1726.37 | 1729.91 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 1721.60 | 1727.49 | 1729.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:15:00 | 1721.60 | 1727.49 | 1729.94 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 1710.25 | 1720.65 | 1726.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 15:15:00 | 1710.25 | 1720.65 | 1726.20 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 1723.90 | 1721.30 | 1725.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:15:00 | 1723.90 | 1721.30 | 1725.99 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 1726.40 | 1722.32 | 1726.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:15:00 | 1726.40 | 1722.32 | 1726.02 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 1721.80 | 1722.22 | 1725.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:15:00 | 1721.80 | 1722.22 | 1725.64 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 1712.15 | 1720.20 | 1724.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:15:00 | 1712.15 | 1720.20 | 1724.41 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 1709.10 | 1701.80 | 1709.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 14:15:00 | 1709.10 | 1701.80 | 1709.20 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 1701.25 | 1701.69 | 1708.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 15:15:00 | 1701.25 | 1701.69 | 1708.48 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1691.20 | 1699.59 | 1706.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:15:00 | 1691.20 | 1699.59 | 1706.91 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1678.45 | 1687.52 | 1696.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 1678.45 | 1687.52 | 1696.07 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 1686.35 | 1683.65 | 1691.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:15:00 | 1686.35 | 1683.65 | 1691.93 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1680.55 | 1683.92 | 1689.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:15:00 | 1680.55 | 1683.92 | 1689.55 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1694.30 | 1686.00 | 1689.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:15:00 | 1694.30 | 1686.00 | 1689.98 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 1684.95 | 1685.79 | 1689.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:15:00 | 1684.95 | 1685.79 | 1689.53 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 1698.80 | 1688.39 | 1690.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:15:00 | 1698.80 | 1688.39 | 1690.37 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 1700.05 | 1690.72 | 1691.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:15:00 | 1700.05 | 1690.72 | 1691.25 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 14:15:00 | 1702.45 | 1693.07 | 1692.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 09:15:00 | 1726.05 | 1701.60 | 1696.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 11:15:00 | 1777.85 | 1785.66 | 1770.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 11:15:00 | 1777.85 | 1785.66 | 1770.92 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1849.65 | 1850.47 | 1831.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 1849.65 | 1850.47 | 1831.16 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 1843.85 | 1860.69 | 1854.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:15:00 | 1843.85 | 1860.69 | 1854.41 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 1848.55 | 1858.26 | 1853.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:15:00 | 1848.55 | 1858.26 | 1853.88 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 14:15:00 | 1840.70 | 1850.66 | 1851.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 09:15:00 | 1826.80 | 1844.33 | 1848.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 09:15:00 | 1894.50 | 1843.04 | 1843.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 09:15:00 | 1894.50 | 1843.04 | 1843.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 1894.50 | 1843.04 | 1843.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 1894.50 | 1843.04 | 1843.09 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-25 10:15:00 | 1850.40 | 1844.51 | 1843.76 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 15:15:00 | 1827.00 | 1842.49 | 1844.11 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 09:15:00 | 1893.00 | 1852.59 | 1848.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 11:15:00 | 1923.80 | 1889.79 | 1876.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 15:15:00 | 1885.00 | 1893.60 | 1883.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-30 15:15:00 | 1885.00 | 1893.60 | 1883.19 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1898.00 | 1894.48 | 1884.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 1898.00 | 1894.48 | 1884.54 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 1885.20 | 1892.62 | 1884.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:15:00 | 1885.20 | 1892.62 | 1884.60 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 1886.75 | 1891.45 | 1884.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:15:00 | 1886.75 | 1891.45 | 1884.79 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 1885.00 | 1890.16 | 1884.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 12:15:00 | 1885.00 | 1890.16 | 1884.81 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 1887.00 | 1889.53 | 1885.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 13:15:00 | 1887.00 | 1889.53 | 1885.01 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 1884.15 | 1888.45 | 1884.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 14:15:00 | 1884.15 | 1888.45 | 1884.93 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 1881.00 | 1886.96 | 1884.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 15:15:00 | 1881.00 | 1886.96 | 1884.58 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 1924.10 | 1902.69 | 1892.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:15:00 | 1924.10 | 1902.69 | 1892.65 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 1906.25 | 1911.53 | 1899.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 13:15:00 | 1906.25 | 1911.53 | 1899.99 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 1892.65 | 1907.75 | 1899.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 14:15:00 | 1892.65 | 1907.75 | 1899.33 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 1895.00 | 1905.20 | 1898.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:15:00 | 1895.00 | 1905.20 | 1898.93 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 1935.25 | 1911.21 | 1902.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:15:00 | 1935.25 | 1911.21 | 1902.24 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 1949.30 | 1980.98 | 1966.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:15:00 | 1949.30 | 1980.98 | 1966.25 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 1943.65 | 1973.51 | 1964.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:15:00 | 1943.65 | 1973.51 | 1964.20 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 1975.30 | 1972.81 | 1965.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:15:00 | 1975.30 | 1972.81 | 1965.43 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 2002.60 | 1979.74 | 1970.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 2002.60 | 1979.74 | 1970.98 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 1964.60 | 1976.71 | 1970.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:15:00 | 1964.60 | 1976.71 | 1970.40 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 1994.25 | 1980.22 | 1972.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:15:00 | 1994.25 | 1980.22 | 1972.57 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 2052.05 | 2047.37 | 2022.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 14:15:00 | 2052.05 | 2047.37 | 2022.94 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 2038.05 | 2058.49 | 2043.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 15:15:00 | 2038.05 | 2058.49 | 2043.94 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 2048.25 | 2056.44 | 2044.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:15:00 | 2048.25 | 2056.44 | 2044.33 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 2045.00 | 2054.15 | 2044.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:15:00 | 2045.00 | 2054.15 | 2044.39 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 2044.05 | 2052.13 | 2044.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 11:15:00 | 2044.05 | 2052.13 | 2044.36 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 2037.75 | 2049.26 | 2043.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 12:15:00 | 2037.75 | 2049.26 | 2043.76 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 2034.05 | 2046.21 | 2042.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 13:15:00 | 2034.05 | 2046.21 | 2042.88 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 2045.00 | 2045.97 | 2043.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:15:00 | 2045.00 | 2045.97 | 2043.07 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 2057.50 | 2048.28 | 2044.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 15:15:00 | 2057.50 | 2048.28 | 2044.38 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 2031.40 | 2044.90 | 2043.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:15:00 | 2031.40 | 2044.90 | 2043.20 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 10:15:00 | 1991.95 | 2034.31 | 2038.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 12:15:00 | 1968.30 | 1998.03 | 2008.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 09:15:00 | 1996.50 | 1981.31 | 1995.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 1996.50 | 1981.31 | 1995.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 1996.50 | 1981.31 | 1995.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:15:00 | 1996.50 | 1981.31 | 1995.60 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 2001.00 | 1985.24 | 1996.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:15:00 | 2001.00 | 1985.24 | 1996.09 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 2007.50 | 1989.70 | 1997.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:15:00 | 2007.50 | 1989.70 | 1997.13 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 15:15:00 | 2025.00 | 2004.83 | 2002.46 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-08-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 15:15:00 | 1994.95 | 2000.71 | 2001.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 1987.00 | 1996.04 | 1998.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 14:15:00 | 1998.20 | 1994.27 | 1997.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 14:15:00 | 1998.20 | 1994.27 | 1997.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 1998.20 | 1994.27 | 1997.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:15:00 | 1998.20 | 1994.27 | 1997.48 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 1999.25 | 1995.27 | 1997.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:15:00 | 1999.25 | 1995.27 | 1997.64 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1982.50 | 1992.71 | 1996.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 1982.50 | 1992.71 | 1996.26 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 2069.00 | 1985.15 | 1987.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 2069.00 | 1985.15 | 1987.02 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 10:15:00 | 2086.40 | 2005.40 | 1996.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 11:15:00 | 2100.95 | 2024.51 | 2005.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-28 09:15:00 | 2139.85 | 2163.32 | 2113.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-28 09:15:00 | 2139.85 | 2163.32 | 2113.27 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 2119.35 | 2142.50 | 2121.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 14:15:00 | 2119.35 | 2142.50 | 2121.13 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 2119.00 | 2137.80 | 2120.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 15:15:00 | 2119.00 | 2137.80 | 2120.94 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 2073.00 | 2124.84 | 2116.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:15:00 | 2073.00 | 2124.84 | 2116.58 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 2121.00 | 2127.27 | 2119.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:15:00 | 2121.00 | 2127.27 | 2119.32 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 2102.50 | 2122.32 | 2117.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 12:15:00 | 2102.50 | 2122.32 | 2117.79 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 2090.55 | 2115.97 | 2115.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 13:15:00 | 2090.55 | 2115.97 | 2115.31 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 15:15:00 | 2101.50 | 2112.95 | 2114.05 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 2162.50 | 2122.86 | 2118.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 10:15:00 | 2204.40 | 2139.17 | 2126.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 09:15:00 | 2238.10 | 2255.66 | 2222.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 09:15:00 | 2238.10 | 2255.66 | 2222.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 2238.10 | 2255.66 | 2222.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 2238.10 | 2255.66 | 2222.63 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 2214.25 | 2247.38 | 2221.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:15:00 | 2214.25 | 2247.38 | 2221.86 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 2194.35 | 2236.77 | 2219.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:15:00 | 2194.35 | 2236.77 | 2219.36 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 09:15:00 | 2184.95 | 2208.71 | 2210.25 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 13:15:00 | 2220.85 | 2211.58 | 2210.78 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 09:15:00 | 2191.70 | 2210.93 | 2211.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 10:15:00 | 2190.00 | 2206.74 | 2209.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 2180.90 | 2142.46 | 2160.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 2180.90 | 2142.46 | 2160.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 2180.90 | 2142.46 | 2160.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:15:00 | 2180.90 | 2142.46 | 2160.46 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 2187.50 | 2151.47 | 2162.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:15:00 | 2187.50 | 2151.47 | 2162.92 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 2177.50 | 2156.67 | 2164.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:15:00 | 2177.50 | 2156.67 | 2164.25 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 2221.20 | 2178.35 | 2172.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 2242.00 | 2191.08 | 2179.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 14:15:00 | 2254.05 | 2255.28 | 2231.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 14:15:00 | 2254.05 | 2255.28 | 2231.69 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 2239.30 | 2249.64 | 2233.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 2239.30 | 2249.64 | 2233.07 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 2226.20 | 2244.95 | 2232.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:15:00 | 2226.20 | 2244.95 | 2232.45 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 2216.50 | 2239.26 | 2231.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:15:00 | 2216.50 | 2239.26 | 2231.00 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 15:15:00 | 2214.65 | 2227.30 | 2227.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 10:15:00 | 2188.80 | 2217.13 | 2222.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 2214.30 | 2197.82 | 2207.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 2214.30 | 2197.82 | 2207.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 2214.30 | 2197.82 | 2207.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 2214.30 | 2197.82 | 2207.87 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 2194.35 | 2197.12 | 2206.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:15:00 | 2194.35 | 2197.12 | 2206.64 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 2191.90 | 2189.72 | 2197.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 10:15:00 | 2191.90 | 2189.72 | 2197.11 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 2195.85 | 2188.74 | 2194.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:15:00 | 2195.85 | 2188.74 | 2194.06 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 2197.20 | 2190.43 | 2194.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 15:15:00 | 2197.20 | 2190.43 | 2194.34 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 09:15:00 | 2224.40 | 2197.23 | 2197.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 10:15:00 | 2254.10 | 2208.60 | 2202.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 13:15:00 | 2195.95 | 2211.11 | 2205.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 13:15:00 | 2195.95 | 2211.11 | 2205.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 2195.95 | 2211.11 | 2205.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:15:00 | 2195.95 | 2211.11 | 2205.57 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 2201.45 | 2209.18 | 2205.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:15:00 | 2201.45 | 2209.18 | 2205.20 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 2204.50 | 2208.24 | 2205.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 15:15:00 | 2204.50 | 2208.24 | 2205.13 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 2228.60 | 2212.31 | 2207.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 2228.60 | 2212.31 | 2207.27 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 2179.15 | 2205.68 | 2204.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:15:00 | 2179.15 | 2205.68 | 2204.71 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 11:15:00 | 2152.00 | 2194.94 | 2199.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 12:15:00 | 2132.60 | 2182.48 | 2193.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 12:15:00 | 2164.55 | 2155.09 | 2170.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-22 12:15:00 | 2164.55 | 2155.09 | 2170.42 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 2140.50 | 2152.17 | 2167.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 13:15:00 | 2140.50 | 2152.17 | 2167.70 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 2149.60 | 2147.17 | 2161.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:15:00 | 2149.60 | 2147.17 | 2161.10 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 2111.20 | 2074.18 | 2100.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:15:00 | 2111.20 | 2074.18 | 2100.58 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 2139.90 | 2087.32 | 2104.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:15:00 | 2139.90 | 2087.32 | 2104.15 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 2136.70 | 2097.20 | 2107.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:15:00 | 2136.70 | 2097.20 | 2107.11 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 2111.75 | 2105.91 | 2109.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:15:00 | 2111.75 | 2105.91 | 2109.24 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 2105.50 | 2105.83 | 2108.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:15:00 | 2105.50 | 2105.83 | 2108.90 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 2104.50 | 2105.56 | 2108.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:15:00 | 2104.50 | 2105.56 | 2108.50 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 2117.35 | 2107.92 | 2109.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:15:00 | 2117.35 | 2107.92 | 2109.31 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 2115.15 | 2109.37 | 2109.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:15:00 | 2115.15 | 2109.37 | 2109.84 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 12:15:00 | 2092.75 | 2106.04 | 2108.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 12:15:00 | 2092.75 | 2106.04 | 2108.28 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 2069.10 | 2096.68 | 2103.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:15:00 | 2069.10 | 2096.68 | 2103.52 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 2048.00 | 2044.57 | 2065.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 2048.00 | 2044.57 | 2065.34 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 2002.35 | 1994.93 | 2013.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 2002.35 | 1994.93 | 2013.16 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 2003.95 | 1982.02 | 1996.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:15:00 | 2003.95 | 1982.02 | 1996.22 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 2011.80 | 1987.97 | 1997.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:15:00 | 2011.80 | 1987.97 | 1997.63 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 2010.00 | 1992.38 | 1998.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:15:00 | 2010.00 | 1992.38 | 1998.76 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 2002.00 | 1993.78 | 1998.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:15:00 | 2002.00 | 1993.78 | 1998.26 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 2009.90 | 1997.00 | 1999.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 14:15:00 | 2009.90 | 1997.00 | 1999.31 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 1999.30 | 1998.23 | 1999.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:15:00 | 1999.30 | 1998.23 | 1999.46 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 2004.05 | 1999.39 | 1999.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:15:00 | 2004.05 | 1999.39 | 1999.88 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 1997.55 | 1999.02 | 1999.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:15:00 | 1997.55 | 1999.02 | 1999.66 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 1994.50 | 1998.12 | 1999.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 13:15:00 | 1994.50 | 1998.12 | 1999.19 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 2005.90 | 1999.67 | 1999.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:15:00 | 2005.90 | 1999.67 | 1999.80 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 1999.50 | 1999.64 | 1999.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 15:15:00 | 1999.50 | 1999.64 | 1999.78 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 09:15:00 | 2040.60 | 2007.83 | 2003.49 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 11:15:00 | 1970.45 | 2006.15 | 2007.94 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 2027.90 | 2008.18 | 2006.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 12:15:00 | 2038.50 | 2014.25 | 2008.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 10:15:00 | 2055.90 | 2055.97 | 2041.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 10:15:00 | 2055.90 | 2055.97 | 2041.02 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 2030.70 | 2050.66 | 2041.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:15:00 | 2030.70 | 2050.66 | 2041.19 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 2020.95 | 2044.72 | 2039.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:15:00 | 2020.95 | 2044.72 | 2039.35 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 15:15:00 | 2015.25 | 2033.79 | 2035.00 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 2048.80 | 2036.79 | 2036.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 10:15:00 | 2073.50 | 2044.13 | 2039.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 13:15:00 | 2039.70 | 2047.05 | 2042.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 13:15:00 | 2039.70 | 2047.05 | 2042.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 2039.70 | 2047.05 | 2042.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:15:00 | 2039.70 | 2047.05 | 2042.53 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 2044.00 | 2046.44 | 2042.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:15:00 | 2044.00 | 2046.44 | 2042.66 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 2042.70 | 2045.69 | 2042.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:15:00 | 2042.70 | 2045.69 | 2042.67 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 2054.40 | 2047.43 | 2043.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 2054.40 | 2047.43 | 2043.73 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 2048.05 | 2047.56 | 2044.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:15:00 | 2048.05 | 2047.56 | 2044.13 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 2052.00 | 2048.44 | 2044.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:15:00 | 2052.00 | 2048.44 | 2044.84 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 2048.15 | 2048.94 | 2045.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 13:15:00 | 2048.15 | 2048.94 | 2045.73 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 2057.05 | 2050.56 | 2046.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:15:00 | 2057.05 | 2050.56 | 2046.75 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 2060.40 | 2058.27 | 2052.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:15:00 | 2060.40 | 2058.27 | 2052.06 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 2040.55 | 2054.73 | 2051.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:15:00 | 2040.55 | 2054.73 | 2051.01 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 2037.00 | 2051.18 | 2049.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:15:00 | 2037.00 | 2051.18 | 2049.74 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 14:15:00 | 2033.10 | 2047.57 | 2048.23 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 2140.00 | 2064.02 | 2055.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 14:15:00 | 2174.45 | 2118.03 | 2088.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 2170.00 | 2171.78 | 2142.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 14:15:00 | 2170.00 | 2171.78 | 2142.40 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 2164.50 | 2171.73 | 2155.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:15:00 | 2164.50 | 2171.73 | 2155.21 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 2289.50 | 2342.09 | 2309.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 2289.50 | 2342.09 | 2309.95 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 2286.55 | 2330.98 | 2307.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:15:00 | 2286.55 | 2330.98 | 2307.82 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 2364.75 | 2366.82 | 2354.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 13:15:00 | 2364.75 | 2366.82 | 2354.83 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 2364.95 | 2366.21 | 2357.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 2364.95 | 2366.21 | 2357.51 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 2409.20 | 2374.81 | 2362.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:15:00 | 2409.20 | 2374.81 | 2362.21 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 2369.15 | 2376.90 | 2365.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 12:15:00 | 2369.15 | 2376.90 | 2365.54 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 2361.95 | 2372.39 | 2366.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 15:15:00 | 2361.95 | 2372.39 | 2366.17 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 2336.60 | 2365.23 | 2363.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:15:00 | 2336.60 | 2365.23 | 2363.48 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 10:15:00 | 2329.30 | 2358.04 | 2360.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 12:15:00 | 2315.50 | 2344.58 | 2353.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 14:15:00 | 2189.10 | 2167.10 | 2213.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 14:15:00 | 2189.10 | 2167.10 | 2213.38 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 2199.90 | 2172.92 | 2207.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:15:00 | 2199.90 | 2172.92 | 2207.95 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 2171.30 | 2172.60 | 2204.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:15:00 | 2171.30 | 2172.60 | 2204.62 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 2200.00 | 2177.37 | 2192.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:15:00 | 2200.00 | 2177.37 | 2192.63 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 2212.45 | 2184.38 | 2194.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:15:00 | 2212.45 | 2184.38 | 2194.43 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 12:15:00 | 2257.05 | 2207.82 | 2203.89 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 2211.00 | 2216.74 | 2217.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 2202.00 | 2212.14 | 2215.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 2234.20 | 2213.47 | 2214.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 2234.20 | 2213.47 | 2214.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 2234.20 | 2213.47 | 2214.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 2234.20 | 2213.47 | 2214.71 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 2245.10 | 2219.79 | 2217.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 12:15:00 | 2265.50 | 2232.56 | 2223.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 2244.60 | 2253.47 | 2238.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 2244.60 | 2253.47 | 2238.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 2244.60 | 2253.47 | 2238.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:15:00 | 2244.60 | 2253.47 | 2238.40 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 2237.95 | 2247.57 | 2239.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:15:00 | 2237.95 | 2247.57 | 2239.17 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 2233.95 | 2244.84 | 2238.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 13:15:00 | 2233.95 | 2244.84 | 2238.70 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 2225.75 | 2237.74 | 2236.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:15:00 | 2225.75 | 2237.74 | 2236.33 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 2249.70 | 2242.29 | 2238.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:15:00 | 2249.70 | 2242.29 | 2238.78 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 2221.30 | 2238.09 | 2237.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:15:00 | 2221.30 | 2238.09 | 2237.19 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 12:15:00 | 2211.65 | 2232.81 | 2234.87 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 2248.45 | 2233.23 | 2232.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 14:15:00 | 2304.60 | 2247.50 | 2238.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 10:15:00 | 2243.50 | 2254.88 | 2245.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 10:15:00 | 2243.50 | 2254.88 | 2245.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 2243.50 | 2254.88 | 2245.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:15:00 | 2243.50 | 2254.88 | 2245.20 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 2247.50 | 2253.40 | 2245.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:15:00 | 2247.50 | 2253.40 | 2245.41 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 12:15:00 | 2243.25 | 2251.37 | 2245.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:15:00 | 2243.25 | 2251.37 | 2245.22 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 2244.20 | 2249.94 | 2245.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 13:15:00 | 2244.20 | 2249.94 | 2245.12 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 2224.10 | 2244.77 | 2243.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 14:15:00 | 2224.10 | 2244.77 | 2243.21 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 15:15:00 | 2227.50 | 2241.32 | 2241.78 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 15:15:00 | 2259.50 | 2243.70 | 2241.70 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 09:15:00 | 2219.65 | 2238.89 | 2239.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 10:15:00 | 2210.50 | 2233.21 | 2237.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 14:15:00 | 2214.15 | 2212.00 | 2219.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 14:15:00 | 2214.15 | 2212.00 | 2219.70 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 2229.05 | 2215.73 | 2220.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:15:00 | 2229.05 | 2215.73 | 2220.08 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 12:15:00 | 2241.25 | 2226.08 | 2224.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 13:15:00 | 2247.30 | 2230.32 | 2226.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 2352.50 | 2369.08 | 2338.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-03 09:15:00 | 2352.50 | 2369.08 | 2338.68 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 2423.60 | 2432.44 | 2409.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:15:00 | 2423.60 | 2432.44 | 2409.53 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 2425.00 | 2430.95 | 2410.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:15:00 | 2425.00 | 2430.95 | 2410.94 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 2407.25 | 2421.14 | 2410.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:15:00 | 2407.25 | 2421.14 | 2410.97 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 2414.50 | 2419.81 | 2411.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:15:00 | 2414.50 | 2419.81 | 2411.29 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 2416.10 | 2419.07 | 2411.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:15:00 | 2416.10 | 2419.07 | 2411.73 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 2353.90 | 2406.03 | 2406.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 2347.60 | 2394.35 | 2401.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 14:15:00 | 2384.30 | 2375.21 | 2388.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 14:15:00 | 2384.30 | 2375.21 | 2388.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 2384.30 | 2375.21 | 2388.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:15:00 | 2384.30 | 2375.21 | 2388.23 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 2247.05 | 2304.84 | 2342.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 14:15:00 | 2247.05 | 2304.84 | 2342.38 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 2266.50 | 2225.35 | 2246.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 2266.50 | 2225.35 | 2246.69 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 2265.85 | 2233.45 | 2248.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:15:00 | 2265.85 | 2233.45 | 2248.43 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 2249.80 | 2239.20 | 2248.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:15:00 | 2249.80 | 2239.20 | 2248.60 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 2248.60 | 2241.08 | 2248.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 13:15:00 | 2248.60 | 2241.08 | 2248.60 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 2247.00 | 2242.26 | 2248.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:15:00 | 2247.00 | 2242.26 | 2248.46 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 15:15:00 | 2246.25 | 2243.06 | 2248.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 15:15:00 | 2246.25 | 2243.06 | 2248.26 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 2252.00 | 2244.85 | 2248.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:15:00 | 2252.00 | 2244.85 | 2248.60 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 10:15:00 | 2285.90 | 2253.06 | 2251.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 11:15:00 | 2295.10 | 2261.47 | 2255.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 2279.50 | 2292.16 | 2275.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:15:00 | 2279.50 | 2292.16 | 2275.77 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 2276.85 | 2289.10 | 2275.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:15:00 | 2276.85 | 2289.10 | 2275.87 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 2267.40 | 2284.76 | 2275.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:15:00 | 2267.40 | 2284.76 | 2275.10 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 2257.05 | 2279.22 | 2273.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:15:00 | 2257.05 | 2279.22 | 2273.46 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 2242.00 | 2271.77 | 2270.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:15:00 | 2242.00 | 2271.77 | 2270.60 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 2230.50 | 2263.52 | 2266.95 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 2262.00 | 2247.69 | 2246.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 2287.95 | 2258.07 | 2251.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 2305.35 | 2305.67 | 2283.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:15:00 | 2305.35 | 2305.67 | 2283.50 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 2365.60 | 2380.77 | 2356.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 2365.60 | 2380.77 | 2356.75 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 2356.65 | 2372.59 | 2356.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:15:00 | 2356.65 | 2372.59 | 2356.99 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 2352.20 | 2368.51 | 2356.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:15:00 | 2352.20 | 2368.51 | 2356.56 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 2348.10 | 2364.43 | 2355.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 13:15:00 | 2348.10 | 2364.43 | 2355.79 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 2340.75 | 2359.70 | 2354.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:15:00 | 2340.75 | 2359.70 | 2354.42 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 2325.10 | 2348.83 | 2350.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 14:15:00 | 2280.15 | 2323.91 | 2336.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 15:15:00 | 2329.95 | 2325.12 | 2335.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 15:15:00 | 2329.95 | 2325.12 | 2335.98 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 2294.25 | 2272.92 | 2294.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:15:00 | 2294.25 | 2272.92 | 2294.86 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 2296.70 | 2277.67 | 2295.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:15:00 | 2296.70 | 2277.67 | 2295.03 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 2304.00 | 2282.94 | 2295.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:15:00 | 2304.00 | 2282.94 | 2295.84 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 2303.35 | 2287.02 | 2296.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:15:00 | 2303.35 | 2287.02 | 2296.53 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 15:15:00 | 2347.65 | 2308.26 | 2305.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 11:15:00 | 2396.35 | 2336.41 | 2319.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 10:15:00 | 2373.05 | 2379.79 | 2353.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 10:15:00 | 2373.05 | 2379.79 | 2353.77 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 2361.10 | 2371.97 | 2356.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:15:00 | 2361.10 | 2371.97 | 2356.33 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 2365.30 | 2367.87 | 2358.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:15:00 | 2365.30 | 2367.87 | 2358.12 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 2395.35 | 2412.54 | 2396.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:15:00 | 2395.35 | 2412.54 | 2396.21 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 2400.00 | 2410.03 | 2396.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 15:15:00 | 2400.00 | 2410.03 | 2396.55 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 2408.00 | 2409.63 | 2397.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 2408.00 | 2409.63 | 2397.59 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 2391.25 | 2405.95 | 2397.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:15:00 | 2391.25 | 2405.95 | 2397.02 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 2394.60 | 2403.68 | 2396.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:15:00 | 2394.60 | 2403.68 | 2396.80 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 2394.00 | 2401.75 | 2396.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:15:00 | 2394.00 | 2401.75 | 2396.54 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 2396.50 | 2400.70 | 2396.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:15:00 | 2396.50 | 2400.70 | 2396.54 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 2386.65 | 2397.89 | 2395.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:15:00 | 2386.65 | 2397.89 | 2395.64 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 2386.05 | 2395.52 | 2394.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 15:15:00 | 2386.05 | 2395.52 | 2394.77 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 2370.90 | 2390.60 | 2392.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 2357.85 | 2374.95 | 2382.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 2298.20 | 2297.67 | 2322.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 15:15:00 | 2298.20 | 2297.67 | 2322.07 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 2303.15 | 2298.77 | 2320.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 2303.15 | 2298.77 | 2320.35 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 2295.00 | 2300.02 | 2317.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:15:00 | 2295.00 | 2300.02 | 2317.29 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 2304.25 | 2298.16 | 2309.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 2304.25 | 2298.16 | 2309.30 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 2304.35 | 2299.40 | 2308.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:15:00 | 2304.35 | 2299.40 | 2308.85 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 2323.15 | 2304.15 | 2310.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:15:00 | 2323.15 | 2304.15 | 2310.15 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 2315.15 | 2306.35 | 2310.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:15:00 | 2315.15 | 2306.35 | 2310.60 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 2293.80 | 2302.71 | 2308.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:15:00 | 2293.80 | 2302.71 | 2308.14 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 2336.95 | 2308.17 | 2309.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:15:00 | 2336.95 | 2308.17 | 2309.59 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 2341.25 | 2314.78 | 2312.47 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 2267.00 | 2310.82 | 2313.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 12:15:00 | 2238.50 | 2282.05 | 2298.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 2220.70 | 2139.19 | 2172.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 2220.70 | 2139.19 | 2172.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 2220.70 | 2139.19 | 2172.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 2220.70 | 2139.19 | 2172.49 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 2179.05 | 2168.18 | 2179.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 12:15:00 | 2179.05 | 2168.18 | 2179.49 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 2170.00 | 2168.54 | 2178.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 13:15:00 | 2170.00 | 2168.54 | 2178.63 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 2175.40 | 2169.91 | 2178.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:15:00 | 2175.40 | 2169.91 | 2178.34 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 2173.95 | 2170.72 | 2177.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 15:15:00 | 2173.95 | 2170.72 | 2177.94 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 2179.75 | 2172.53 | 2178.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:15:00 | 2179.75 | 2172.53 | 2178.10 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 2180.00 | 2174.02 | 2178.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:15:00 | 2180.00 | 2174.02 | 2178.27 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 2160.40 | 2171.30 | 2176.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 11:15:00 | 2160.40 | 2171.30 | 2176.65 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 2162.95 | 2166.20 | 2172.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:15:00 | 2162.95 | 2166.20 | 2172.73 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 13:15:00 | 2162.55 | 2153.05 | 2161.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 13:15:00 | 2162.55 | 2153.05 | 2161.63 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 2197.00 | 2161.84 | 2164.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 14:15:00 | 2197.00 | 2161.84 | 2164.85 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 2230.00 | 2175.47 | 2170.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 09:15:00 | 2286.40 | 2219.76 | 2199.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 2295.00 | 2302.70 | 2262.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 09:15:00 | 2295.00 | 2302.70 | 2262.06 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 2371.50 | 2326.68 | 2294.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 2371.50 | 2326.68 | 2294.39 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 2322.50 | 2418.66 | 2388.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 09:15:00 | 2322.50 | 2418.66 | 2388.00 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 11:15:00 | 2247.50 | 2357.07 | 2363.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 2173.60 | 2238.95 | 2277.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 2147.35 | 2144.19 | 2201.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 09:15:00 | 2147.35 | 2144.19 | 2201.46 | Sideways: bar.high >= retest1.high within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 1901.75 | 1847.19 | 1867.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 1901.75 | 1847.19 | 1867.41 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 1834.05 | 1844.56 | 1864.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:15:00 | 1834.05 | 1844.56 | 1864.38 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 1830.00 | 1810.35 | 1824.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:15:00 | 1830.00 | 1810.35 | 1824.78 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 1832.05 | 1814.69 | 1825.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 1832.05 | 1814.69 | 1825.44 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 1825.45 | 1816.84 | 1825.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:15:00 | 1825.45 | 1816.84 | 1825.44 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 1806.95 | 1814.86 | 1823.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:15:00 | 1806.95 | 1814.86 | 1823.76 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 1663.00 | 1662.08 | 1690.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:15:00 | 1663.00 | 1662.08 | 1690.65 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 1672.80 | 1664.22 | 1689.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:15:00 | 1672.80 | 1664.22 | 1689.02 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 13:15:00 | 1623.85 | 1584.83 | 1597.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 13:15:00 | 1623.85 | 1584.83 | 1597.85 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 14:15:00 | 1608.15 | 1589.49 | 1598.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 14:15:00 | 1608.15 | 1589.49 | 1598.79 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 1558.00 | 1563.17 | 1578.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 1558.00 | 1563.17 | 1578.03 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 1542.50 | 1512.64 | 1526.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:15:00 | 1542.50 | 1512.64 | 1526.64 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 1549.65 | 1520.04 | 1528.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:15:00 | 1549.65 | 1520.04 | 1528.73 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 1572.90 | 1538.25 | 1535.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 1609.75 | 1565.30 | 1550.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 1559.85 | 1571.89 | 1560.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 1559.85 | 1571.89 | 1560.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 1559.85 | 1571.89 | 1560.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:15:00 | 1559.85 | 1571.89 | 1560.46 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 1559.00 | 1569.31 | 1560.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:15:00 | 1559.00 | 1569.31 | 1560.33 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1529.50 | 1561.35 | 1557.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 1529.50 | 1561.35 | 1557.53 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 1504.00 | 1562.50 | 1562.74 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 15:15:00 | 1604.00 | 1556.14 | 1555.70 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-03-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 09:15:00 | 1512.30 | 1547.37 | 1551.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 1504.70 | 1538.84 | 1547.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 1533.10 | 1530.06 | 1539.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 1533.10 | 1530.06 | 1539.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1533.10 | 1530.06 | 1539.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:15:00 | 1533.10 | 1530.06 | 1539.84 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1509.00 | 1497.45 | 1513.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:15:00 | 1509.00 | 1497.45 | 1513.74 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 1469.10 | 1478.79 | 1491.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:15:00 | 1469.10 | 1478.79 | 1491.78 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1479.10 | 1473.65 | 1485.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:15:00 | 1479.10 | 1473.65 | 1485.67 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 1476.20 | 1474.93 | 1484.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:15:00 | 1476.20 | 1474.93 | 1484.21 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 1463.90 | 1472.07 | 1480.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 14:15:00 | 1463.90 | 1472.07 | 1480.56 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 1408.00 | 1425.59 | 1448.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:15:00 | 1408.00 | 1425.59 | 1448.12 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1461.20 | 1430.35 | 1444.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:15:00 | 1461.20 | 1430.35 | 1444.37 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1469.90 | 1438.26 | 1446.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:15:00 | 1469.90 | 1438.26 | 1446.69 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1482.60 | 1453.68 | 1452.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 1498.00 | 1462.54 | 1456.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1432.40 | 1476.09 | 1469.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1432.40 | 1476.09 | 1469.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1432.40 | 1476.09 | 1469.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 1432.40 | 1476.09 | 1469.02 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 1425.00 | 1465.87 | 1465.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:15:00 | 1425.00 | 1465.87 | 1465.02 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 1473.80 | 1467.40 | 1465.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:15:00 | 1473.80 | 1467.40 | 1465.86 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 1448.50 | 1463.62 | 1464.28 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 14:15:00 | 1470.00 | 1464.89 | 1464.80 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 1456.10 | 1463.14 | 1464.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1407.00 | 1451.91 | 1458.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1446.90 | 1416.01 | 1431.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1446.90 | 1416.01 | 1431.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1446.90 | 1416.01 | 1431.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 1446.90 | 1416.01 | 1431.86 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1414.30 | 1415.67 | 1430.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:15:00 | 1414.30 | 1415.67 | 1430.26 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 1431.00 | 1418.73 | 1430.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:15:00 | 1431.00 | 1418.73 | 1430.33 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 1430.00 | 1420.99 | 1430.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 12:15:00 | 1430.00 | 1420.99 | 1430.30 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 13:15:00 | 1465.00 | 1429.79 | 1433.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 13:15:00 | 1465.00 | 1429.79 | 1433.45 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 1486.10 | 1441.05 | 1438.24 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 1418.80 | 1436.84 | 1437.39 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 1449.10 | 1437.92 | 1437.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 09:15:00 | 1458.30 | 1442.65 | 1439.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 15:15:00 | 1495.20 | 1496.37 | 1481.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 15:15:00 | 1495.20 | 1496.37 | 1481.75 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 1494.60 | 1528.29 | 1512.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 09:15:00 | 1494.60 | 1528.29 | 1512.20 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 1499.60 | 1522.55 | 1511.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:15:00 | 1499.60 | 1522.55 | 1511.06 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-04-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 15:15:00 | 1495.90 | 1506.16 | 1506.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 09:15:00 | 1471.80 | 1493.37 | 1498.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 1498.80 | 1472.31 | 1482.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 1498.80 | 1472.31 | 1482.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1498.80 | 1472.31 | 1482.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 09:15:00 | 1498.80 | 1472.31 | 1482.45 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 10:15:00 | 1487.10 | 1475.27 | 1482.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 10:15:00 | 1487.10 | 1475.27 | 1482.87 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 12:15:00 | 1484.00 | 1478.63 | 1483.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 12:15:00 | 1484.00 | 1478.63 | 1483.18 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 13:15:00 | 1494.40 | 1481.79 | 1484.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 13:15:00 | 1494.40 | 1481.79 | 1484.20 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 1487.40 | 1482.91 | 1484.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 14:15:00 | 1487.40 | 1482.91 | 1484.49 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2026-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 09:15:00 | 1555.60 | 1498.44 | 1491.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 1607.80 | 1555.93 | 1529.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 1600.00 | 1610.12 | 1576.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 09:15:00 | 1600.00 | 1610.12 | 1576.96 | Sideways: bar.low <= retest1.low within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 1583.60 | 1604.14 | 1587.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 11:15:00 | 1583.60 | 1604.14 | 1587.21 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 1576.30 | 1598.58 | 1586.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 12:15:00 | 1576.30 | 1598.58 | 1586.22 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 1572.20 | 1593.30 | 1584.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 13:15:00 | 1572.20 | 1593.30 | 1584.94 | Sideways: bar.low <= retest2.low within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 1524.60 | 1572.00 | 1576.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 10:15:00 | 1484.30 | 1554.46 | 1568.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1496.80 | 1474.22 | 1489.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1496.80 | 1474.22 | 1489.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1496.80 | 1474.22 | 1489.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 1496.80 | 1474.22 | 1489.80 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1485.10 | 1476.40 | 1489.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:15:00 | 1485.10 | 1476.40 | 1489.38 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 1489.20 | 1478.96 | 1489.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:15:00 | 1489.20 | 1478.96 | 1489.36 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 1486.10 | 1480.39 | 1489.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:15:00 | 1486.10 | 1480.39 | 1489.06 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 1468.00 | 1478.38 | 1485.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:15:00 | 1468.00 | 1478.38 | 1485.46 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 1457.10 | 1457.43 | 1469.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:15:00 | 1457.10 | 1457.43 | 1469.13 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 1442.20 | 1427.90 | 1443.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:15:00 | 1442.20 | 1427.90 | 1443.44 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 1435.50 | 1429.42 | 1442.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:15:00 | 1435.50 | 1429.42 | 1442.72 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1426.20 | 1428.11 | 1437.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 1426.20 | 1428.11 | 1437.87 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 15:15:00 | 1419.30 | 1413.77 | 1424.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 15:15:00 | 1419.30 | 1413.77 | 1424.61 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 1460.30 | 1423.08 | 1427.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:15:00 | 1460.30 | 1423.08 | 1427.86 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 1436.50 | 1425.76 | 1428.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:15:00 | 1436.50 | 1425.76 | 1428.64 | Sideways: bar.high >= retest2.high within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 1447.70 | 1433.56 | 1431.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 1499.20 | 1452.22 | 1441.56 | Break + close above crossover candle high |

