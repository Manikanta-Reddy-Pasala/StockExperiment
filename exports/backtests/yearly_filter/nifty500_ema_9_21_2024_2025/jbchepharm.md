# J.B. Chemicals & Pharmaceuticals Ltd. (JBCHEPHARM)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 2155.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 152 |
| ALERT1 | 110 |
| ALERT2 | 109 |
| ALERT2_SKIP | 56 |
| ALERT3 | 304 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 9 |
| ENTRY2 | 123 |
| PARTIAL | 14 |
| TARGET_HIT | 8 |
| STOP_HIT | 120 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 142 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 44 / 98
- **Target hits / Stop hits / Partials:** 8 / 120 / 14
- **Avg / median % per leg:** 0.32% / -0.85%
- **Sum % (uncompounded):** 46.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 61 | 11 | 18.0% | 0 | 61 | 0 | -0.56% | -34.5% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.97% | -2.9% |
| BUY @ 3rd Alert (retest2) | 58 | 11 | 19.0% | 0 | 58 | 0 | -0.54% | -31.5% |
| SELL (all) | 81 | 33 | 40.7% | 8 | 59 | 14 | 0.99% | 80.6% |
| SELL @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 0 | 6 | 0 | 0.21% | 1.2% |
| SELL @ 3rd Alert (retest2) | 75 | 29 | 38.7% | 8 | 53 | 14 | 1.06% | 79.3% |
| retest1 (combined) | 9 | 4 | 44.4% | 0 | 9 | 0 | -0.19% | -1.7% |
| retest2 (combined) | 133 | 40 | 30.1% | 8 | 111 | 14 | 0.36% | 47.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 09:15:00 | 1794.20 | 1798.37 | 1798.39 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 10:15:00 | 1802.80 | 1799.26 | 1798.79 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 11:15:00 | 1793.35 | 1798.08 | 1798.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-14 12:15:00 | 1786.10 | 1794.58 | 1796.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-14 13:15:00 | 1794.90 | 1794.64 | 1796.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-14 13:15:00 | 1794.90 | 1794.64 | 1796.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 13:15:00 | 1794.90 | 1794.64 | 1796.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 13:30:00 | 1797.15 | 1794.64 | 1796.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 14:15:00 | 1799.40 | 1795.60 | 1796.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 15:00:00 | 1799.40 | 1795.60 | 1796.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 15:15:00 | 1795.00 | 1795.48 | 1796.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-15 09:15:00 | 1803.90 | 1795.48 | 1796.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 1829.70 | 1802.32 | 1799.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-18 09:15:00 | 1858.85 | 1827.57 | 1819.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 1792.00 | 1825.09 | 1820.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 1792.00 | 1825.09 | 1820.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 1792.00 | 1825.09 | 1820.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:00:00 | 1792.00 | 1825.09 | 1820.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 1793.00 | 1818.67 | 1818.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:45:00 | 1793.20 | 1818.67 | 1818.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2024-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 11:15:00 | 1805.10 | 1815.96 | 1817.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 14:15:00 | 1773.50 | 1803.78 | 1810.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 15:15:00 | 1675.00 | 1673.53 | 1696.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-27 09:15:00 | 1680.10 | 1673.53 | 1696.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 1741.60 | 1687.15 | 1700.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 10:00:00 | 1741.60 | 1687.15 | 1700.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 1749.00 | 1699.52 | 1704.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 10:30:00 | 1750.00 | 1699.52 | 1704.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2024-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 11:15:00 | 1789.90 | 1717.59 | 1712.52 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 11:15:00 | 1713.60 | 1751.26 | 1753.95 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 14:15:00 | 1770.40 | 1751.66 | 1750.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 15:15:00 | 1816.00 | 1764.53 | 1756.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-03 10:15:00 | 1755.25 | 1764.50 | 1757.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 10:15:00 | 1755.25 | 1764.50 | 1757.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 10:15:00 | 1755.25 | 1764.50 | 1757.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-03 11:00:00 | 1755.25 | 1764.50 | 1757.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 11:15:00 | 1758.05 | 1763.21 | 1757.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 15:00:00 | 1764.00 | 1761.36 | 1758.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 09:15:00 | 1730.75 | 1761.26 | 1758.92 | SL hit (close<static) qty=1.00 sl=1750.00 alert=retest2 |

### Cycle 9 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 1676.60 | 1744.33 | 1751.44 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 13:15:00 | 1781.95 | 1752.64 | 1749.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 10:15:00 | 1810.25 | 1780.34 | 1769.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 13:15:00 | 1885.80 | 1895.45 | 1884.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-13 13:15:00 | 1885.80 | 1895.45 | 1884.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 13:15:00 | 1885.80 | 1895.45 | 1884.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 14:00:00 | 1885.80 | 1895.45 | 1884.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 14:15:00 | 1885.00 | 1893.36 | 1884.91 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2024-06-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 12:15:00 | 1866.55 | 1881.39 | 1882.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 14:15:00 | 1858.75 | 1874.35 | 1878.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-19 11:15:00 | 1825.25 | 1815.51 | 1835.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-19 12:00:00 | 1825.25 | 1815.51 | 1835.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 1836.20 | 1817.75 | 1828.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:00:00 | 1836.20 | 1817.75 | 1828.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 1837.20 | 1821.64 | 1829.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 11:00:00 | 1837.20 | 1821.64 | 1829.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 11:15:00 | 1833.05 | 1823.92 | 1829.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 11:30:00 | 1843.45 | 1823.92 | 1829.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 1800.00 | 1811.73 | 1821.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 11:30:00 | 1773.00 | 1801.11 | 1814.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 13:15:00 | 1684.35 | 1711.21 | 1730.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-28 09:15:00 | 1757.00 | 1712.83 | 1725.89 | SL hit (close>ema200) qty=0.50 sl=1712.83 alert=retest2 |

### Cycle 12 — BUY (started 2024-06-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 12:15:00 | 1759.00 | 1733.56 | 1733.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 13:15:00 | 1766.00 | 1740.05 | 1736.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 14:15:00 | 1786.40 | 1794.11 | 1781.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 15:00:00 | 1786.40 | 1794.11 | 1781.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 15:15:00 | 1800.00 | 1795.29 | 1783.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 10:00:00 | 1804.80 | 1797.19 | 1785.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 11:30:00 | 1805.40 | 1796.64 | 1786.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 12:15:00 | 1804.00 | 1796.64 | 1786.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 15:00:00 | 1803.95 | 1797.13 | 1789.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 1792.50 | 1796.74 | 1790.62 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-05 09:15:00 | 1753.40 | 1782.71 | 1786.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2024-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 09:15:00 | 1753.40 | 1782.71 | 1786.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 12:15:00 | 1740.75 | 1764.37 | 1775.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 11:15:00 | 1750.95 | 1750.06 | 1762.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-08 12:00:00 | 1750.95 | 1750.06 | 1762.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 1710.10 | 1725.65 | 1738.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 10:15:00 | 1701.45 | 1725.65 | 1738.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 11:15:00 | 1706.85 | 1722.41 | 1735.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 09:15:00 | 1744.75 | 1727.25 | 1729.02 | SL hit (close>static) qty=1.00 sl=1740.00 alert=retest2 |

### Cycle 14 — BUY (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 10:15:00 | 1747.65 | 1731.33 | 1730.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 11:15:00 | 1757.60 | 1736.58 | 1733.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 1801.05 | 1802.14 | 1792.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-19 10:15:00 | 1806.90 | 1802.14 | 1792.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 11:15:00 | 1781.25 | 1797.78 | 1791.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:45:00 | 1776.90 | 1797.78 | 1791.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 12:15:00 | 1794.55 | 1797.13 | 1792.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 09:30:00 | 1807.95 | 1798.79 | 1794.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 12:30:00 | 1816.70 | 1815.43 | 1807.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 14:15:00 | 1912.15 | 1916.63 | 1917.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2024-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 14:15:00 | 1912.15 | 1916.63 | 1917.01 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2024-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-02 10:15:00 | 1944.70 | 1922.49 | 1919.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-05 09:15:00 | 1966.15 | 1940.61 | 1931.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 10:15:00 | 1925.00 | 1937.49 | 1930.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 10:15:00 | 1925.00 | 1937.49 | 1930.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 10:15:00 | 1925.00 | 1937.49 | 1930.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 11:00:00 | 1925.00 | 1937.49 | 1930.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 11:15:00 | 1939.95 | 1937.98 | 1931.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 11:30:00 | 1924.70 | 1937.98 | 1931.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 12:15:00 | 1923.75 | 1935.14 | 1930.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 12:45:00 | 1921.70 | 1935.14 | 1930.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 13:15:00 | 1927.00 | 1933.51 | 1930.48 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2024-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 14:15:00 | 1904.60 | 1927.73 | 1928.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 13:15:00 | 1902.00 | 1916.27 | 1921.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 10:15:00 | 1925.15 | 1909.14 | 1915.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 10:15:00 | 1925.15 | 1909.14 | 1915.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 1925.15 | 1909.14 | 1915.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:30:00 | 1927.80 | 1909.14 | 1915.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 1915.95 | 1910.50 | 1915.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 15:15:00 | 1910.05 | 1914.26 | 1916.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 09:15:00 | 1942.70 | 1919.27 | 1917.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 09:15:00 | 1942.70 | 1919.27 | 1917.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-13 09:15:00 | 1962.95 | 1944.08 | 1938.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 11:15:00 | 1943.80 | 1945.79 | 1940.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 11:15:00 | 1943.80 | 1945.79 | 1940.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 11:15:00 | 1943.80 | 1945.79 | 1940.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:00:00 | 1943.80 | 1945.79 | 1940.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 1953.15 | 1947.26 | 1941.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:30:00 | 1927.65 | 1947.26 | 1941.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 1945.50 | 1946.55 | 1942.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 15:15:00 | 1933.05 | 1946.55 | 1942.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 1933.05 | 1943.85 | 1941.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:15:00 | 1937.90 | 1943.85 | 1941.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 1938.50 | 1942.78 | 1941.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 1923.00 | 1942.78 | 1941.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 1937.80 | 1941.78 | 1940.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 10:45:00 | 1937.10 | 1941.78 | 1940.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 11:15:00 | 1964.40 | 1946.31 | 1942.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 12:15:00 | 1967.95 | 1946.31 | 1942.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 12:45:00 | 1988.75 | 1951.50 | 1945.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 09:15:00 | 2004.60 | 1959.48 | 1951.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-19 11:45:00 | 1967.35 | 1967.90 | 1964.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 12:15:00 | 1969.45 | 1968.21 | 1965.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 12:30:00 | 1966.15 | 1968.21 | 1965.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 14:15:00 | 1947.40 | 1964.35 | 1963.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 14:45:00 | 1933.95 | 1964.35 | 1963.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-19 15:15:00 | 1950.20 | 1961.52 | 1962.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2024-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 15:15:00 | 1950.20 | 1961.52 | 1962.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 09:15:00 | 1939.95 | 1957.21 | 1960.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 15:15:00 | 1950.00 | 1946.32 | 1952.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-21 09:15:00 | 1936.35 | 1946.32 | 1952.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 1942.65 | 1945.59 | 1951.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:30:00 | 1943.40 | 1945.59 | 1951.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 1935.15 | 1933.70 | 1941.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-23 09:15:00 | 1930.20 | 1936.14 | 1939.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 11:00:00 | 1899.45 | 1902.87 | 1916.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 10:15:00 | 1942.75 | 1916.15 | 1915.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2024-08-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 10:15:00 | 1942.75 | 1916.15 | 1915.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 14:15:00 | 1960.00 | 1935.98 | 1926.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-29 11:15:00 | 1963.60 | 1966.03 | 1954.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-29 12:00:00 | 1963.60 | 1966.03 | 1954.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 1956.90 | 1963.85 | 1957.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:30:00 | 1946.45 | 1962.00 | 1956.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 1940.00 | 1957.60 | 1955.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:00:00 | 1940.00 | 1957.60 | 1955.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 1945.30 | 1955.14 | 1954.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 12:15:00 | 1940.10 | 1955.14 | 1954.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2024-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 12:15:00 | 1940.00 | 1952.11 | 1953.13 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2024-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 09:15:00 | 1966.95 | 1954.67 | 1954.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 12:15:00 | 1972.85 | 1962.42 | 1958.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 11:15:00 | 1964.95 | 1971.42 | 1965.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 11:15:00 | 1964.95 | 1971.42 | 1965.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 11:15:00 | 1964.95 | 1971.42 | 1965.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 12:00:00 | 1964.95 | 1971.42 | 1965.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 12:15:00 | 1954.50 | 1968.04 | 1964.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 12:30:00 | 1953.75 | 1968.04 | 1964.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 13:15:00 | 1947.50 | 1963.93 | 1962.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 13:30:00 | 1942.30 | 1963.93 | 1962.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2024-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 14:15:00 | 1926.20 | 1956.39 | 1959.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 15:15:00 | 1921.00 | 1930.32 | 1936.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 10:15:00 | 1928.70 | 1928.67 | 1934.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 11:00:00 | 1928.70 | 1928.67 | 1934.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 11:15:00 | 1925.65 | 1928.07 | 1933.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 11:30:00 | 1933.70 | 1928.07 | 1933.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 1929.50 | 1928.35 | 1933.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 13:45:00 | 1930.35 | 1928.35 | 1933.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 1938.90 | 1930.46 | 1933.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 15:00:00 | 1938.90 | 1930.46 | 1933.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 1943.00 | 1932.97 | 1934.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:15:00 | 1963.00 | 1932.97 | 1934.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 1935.55 | 1933.49 | 1934.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 13:00:00 | 1917.95 | 1929.89 | 1932.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 13:30:00 | 1920.90 | 1926.70 | 1930.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 14:00:00 | 1913.95 | 1926.70 | 1930.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 11:15:00 | 1916.60 | 1924.63 | 1928.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 11:15:00 | 1906.10 | 1920.92 | 1926.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 12:45:00 | 1896.60 | 1915.53 | 1923.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 12:15:00 | 1921.45 | 1901.00 | 1900.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2024-09-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 12:15:00 | 1921.45 | 1901.00 | 1900.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 14:15:00 | 1924.30 | 1909.11 | 1904.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 09:15:00 | 1906.95 | 1911.38 | 1906.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 09:15:00 | 1906.95 | 1911.38 | 1906.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 1906.95 | 1911.38 | 1906.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 10:00:00 | 1906.95 | 1911.38 | 1906.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 1906.15 | 1910.34 | 1906.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 11:00:00 | 1906.15 | 1910.34 | 1906.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 1898.35 | 1907.94 | 1905.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 11:30:00 | 1893.90 | 1907.94 | 1905.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 1903.05 | 1906.96 | 1905.28 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2024-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 14:15:00 | 1895.60 | 1903.85 | 1904.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 09:15:00 | 1880.00 | 1899.11 | 1901.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 09:15:00 | 1872.60 | 1866.08 | 1877.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 09:15:00 | 1872.60 | 1866.08 | 1877.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 1872.60 | 1866.08 | 1877.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:00:00 | 1872.60 | 1866.08 | 1877.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 1886.10 | 1870.09 | 1877.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:45:00 | 1890.00 | 1870.09 | 1877.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 1870.25 | 1870.12 | 1877.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 13:15:00 | 1854.00 | 1868.10 | 1875.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 09:15:00 | 1903.95 | 1876.16 | 1876.73 | SL hit (close>static) qty=1.00 sl=1887.45 alert=retest2 |

### Cycle 26 — BUY (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 10:15:00 | 1886.45 | 1878.22 | 1877.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 09:15:00 | 1943.65 | 1899.19 | 1888.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 15:15:00 | 1930.80 | 1939.39 | 1924.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 09:15:00 | 1901.60 | 1939.39 | 1924.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 1888.95 | 1929.30 | 1921.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 1888.95 | 1929.30 | 1921.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2024-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 10:15:00 | 1859.95 | 1915.43 | 1916.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 14:15:00 | 1831.00 | 1870.64 | 1886.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 09:15:00 | 1890.55 | 1869.72 | 1883.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 09:15:00 | 1890.55 | 1869.72 | 1883.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 1890.55 | 1869.72 | 1883.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:00:00 | 1890.55 | 1869.72 | 1883.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 1892.00 | 1874.17 | 1884.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 12:00:00 | 1880.10 | 1875.36 | 1883.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 10:15:00 | 1786.09 | 1826.48 | 1848.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-04 09:15:00 | 1692.09 | 1763.58 | 1803.02 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 28 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 1780.00 | 1738.15 | 1735.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 11:15:00 | 1809.70 | 1759.55 | 1746.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 09:15:00 | 1801.25 | 1805.34 | 1788.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 10:00:00 | 1801.25 | 1805.34 | 1788.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 1808.65 | 1809.11 | 1799.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:30:00 | 1802.55 | 1809.11 | 1799.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 13:15:00 | 1865.45 | 1871.20 | 1854.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 14:30:00 | 1874.05 | 1877.80 | 1858.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 15:00:00 | 1904.20 | 1877.80 | 1858.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 09:30:00 | 1881.95 | 1885.50 | 1876.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-22 13:15:00 | 1879.10 | 1889.72 | 1889.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2024-10-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 13:15:00 | 1879.10 | 1889.72 | 1889.84 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2024-10-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-22 14:15:00 | 1904.95 | 1892.76 | 1891.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-23 09:15:00 | 1915.90 | 1896.17 | 1892.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-24 13:15:00 | 1908.05 | 1914.45 | 1908.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-24 13:15:00 | 1908.05 | 1914.45 | 1908.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 13:15:00 | 1908.05 | 1914.45 | 1908.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-24 13:30:00 | 1903.95 | 1914.45 | 1908.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 14:15:00 | 1902.00 | 1911.96 | 1907.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-24 15:00:00 | 1902.00 | 1911.96 | 1907.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 15:15:00 | 1883.20 | 1906.21 | 1905.59 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 1891.00 | 1903.16 | 1904.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 12:15:00 | 1863.35 | 1892.33 | 1898.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-25 14:15:00 | 1893.15 | 1888.54 | 1895.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-25 14:15:00 | 1893.15 | 1888.54 | 1895.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 14:15:00 | 1893.15 | 1888.54 | 1895.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-25 15:00:00 | 1893.15 | 1888.54 | 1895.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 15:15:00 | 1882.05 | 1887.24 | 1894.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 09:30:00 | 1871.80 | 1885.10 | 1892.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 10:15:00 | 1910.40 | 1890.16 | 1894.42 | SL hit (close>static) qty=1.00 sl=1895.00 alert=retest2 |

### Cycle 32 — BUY (started 2024-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 09:15:00 | 1927.20 | 1880.70 | 1876.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 14:15:00 | 1976.00 | 1917.11 | 1897.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 1881.15 | 1922.17 | 1907.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 1881.15 | 1922.17 | 1907.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1881.15 | 1922.17 | 1907.44 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 12:15:00 | 1864.95 | 1897.36 | 1898.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 14:15:00 | 1852.70 | 1883.25 | 1891.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 09:15:00 | 1877.75 | 1876.12 | 1886.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 10:00:00 | 1877.75 | 1876.12 | 1886.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 1874.95 | 1875.88 | 1885.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 11:15:00 | 1872.90 | 1875.88 | 1885.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-06 14:00:00 | 1870.10 | 1861.17 | 1869.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-06 14:45:00 | 1867.05 | 1865.44 | 1870.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 09:15:00 | 1872.55 | 1870.36 | 1872.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 1871.55 | 1872.14 | 1872.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 11:15:00 | 1867.90 | 1872.14 | 1872.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 12:00:00 | 1869.05 | 1871.52 | 1872.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 09:15:00 | 1779.26 | 1829.11 | 1846.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 09:15:00 | 1776.59 | 1829.11 | 1846.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 09:15:00 | 1773.70 | 1829.11 | 1846.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 09:15:00 | 1778.92 | 1829.11 | 1846.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 09:15:00 | 1774.51 | 1829.11 | 1846.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 09:15:00 | 1775.60 | 1829.11 | 1846.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-11-14 09:15:00 | 1685.61 | 1724.50 | 1755.91 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 34 — BUY (started 2024-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-21 15:15:00 | 1710.00 | 1694.57 | 1694.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 09:15:00 | 1723.85 | 1700.42 | 1696.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 14:15:00 | 1754.75 | 1758.96 | 1741.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 15:00:00 | 1754.75 | 1758.96 | 1741.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 1773.90 | 1760.91 | 1745.04 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 10:15:00 | 1729.00 | 1742.82 | 1744.01 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2024-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 11:15:00 | 1754.50 | 1745.16 | 1744.97 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2024-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 13:15:00 | 1728.75 | 1741.78 | 1743.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 15:15:00 | 1724.95 | 1736.18 | 1740.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 12:15:00 | 1736.95 | 1734.79 | 1738.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-29 13:00:00 | 1736.95 | 1734.79 | 1738.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 1735.05 | 1734.84 | 1737.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 13:30:00 | 1737.00 | 1734.84 | 1737.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 1736.95 | 1735.26 | 1737.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 15:00:00 | 1736.95 | 1735.26 | 1737.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 1734.60 | 1735.13 | 1737.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:15:00 | 1747.10 | 1735.13 | 1737.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2024-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 09:15:00 | 1763.65 | 1740.83 | 1739.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 09:15:00 | 1770.25 | 1753.22 | 1747.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 09:15:00 | 1801.80 | 1803.59 | 1787.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 10:00:00 | 1801.80 | 1803.59 | 1787.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 1790.05 | 1800.88 | 1787.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:00:00 | 1790.05 | 1800.88 | 1787.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 1795.40 | 1799.78 | 1788.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:30:00 | 1789.20 | 1799.78 | 1788.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 12:15:00 | 1784.75 | 1796.78 | 1788.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 12:45:00 | 1785.00 | 1796.78 | 1788.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 13:15:00 | 1783.95 | 1794.21 | 1787.75 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 09:15:00 | 1769.05 | 1783.95 | 1784.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 12:15:00 | 1759.45 | 1775.07 | 1779.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 10:15:00 | 1769.40 | 1767.16 | 1773.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-09 10:15:00 | 1769.40 | 1767.16 | 1773.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 1769.40 | 1767.16 | 1773.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:45:00 | 1768.95 | 1767.16 | 1773.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 11:15:00 | 1764.95 | 1766.72 | 1772.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 12:00:00 | 1764.95 | 1766.72 | 1772.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 13:15:00 | 1786.70 | 1771.02 | 1773.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 13:45:00 | 1795.70 | 1771.02 | 1773.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 14:15:00 | 1782.00 | 1773.22 | 1774.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 14:45:00 | 1787.80 | 1773.22 | 1774.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 09:15:00 | 1784.50 | 1776.56 | 1775.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 09:15:00 | 1808.80 | 1795.94 | 1787.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-16 14:15:00 | 1822.75 | 1842.59 | 1835.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 14:15:00 | 1822.75 | 1842.59 | 1835.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 14:15:00 | 1822.75 | 1842.59 | 1835.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 15:00:00 | 1822.75 | 1842.59 | 1835.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 15:15:00 | 1822.00 | 1838.47 | 1834.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:15:00 | 1828.60 | 1838.47 | 1834.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 1829.75 | 1836.73 | 1833.83 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2024-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 12:15:00 | 1825.00 | 1831.17 | 1831.72 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2024-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 09:15:00 | 1841.05 | 1832.16 | 1831.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-18 14:15:00 | 1856.60 | 1845.96 | 1839.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 10:15:00 | 1896.00 | 1902.78 | 1882.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-20 11:00:00 | 1896.00 | 1902.78 | 1882.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 1902.75 | 1902.77 | 1884.52 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2024-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 09:15:00 | 1833.25 | 1877.48 | 1878.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 11:15:00 | 1809.05 | 1835.48 | 1846.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 09:15:00 | 1823.45 | 1820.90 | 1833.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 09:15:00 | 1823.45 | 1820.90 | 1833.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 1823.45 | 1820.90 | 1833.87 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 14:15:00 | 1848.05 | 1825.29 | 1825.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 10:15:00 | 1877.15 | 1845.44 | 1835.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 12:15:00 | 1840.60 | 1845.51 | 1837.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 12:15:00 | 1840.60 | 1845.51 | 1837.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 12:15:00 | 1840.60 | 1845.51 | 1837.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 13:00:00 | 1840.60 | 1845.51 | 1837.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 1846.75 | 1846.47 | 1839.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 15:00:00 | 1846.75 | 1846.47 | 1839.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 15:15:00 | 1833.45 | 1843.87 | 1838.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 09:30:00 | 1851.25 | 1845.97 | 1840.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 10:00:00 | 1854.40 | 1845.97 | 1840.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 13:45:00 | 1850.75 | 1850.29 | 1844.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 14:45:00 | 1870.75 | 1858.46 | 1848.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 15:15:00 | 1858.85 | 1858.54 | 1849.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 09:15:00 | 1850.20 | 1858.54 | 1849.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 1838.95 | 1854.62 | 1848.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:00:00 | 1838.95 | 1854.62 | 1848.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 1843.20 | 1852.34 | 1848.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:30:00 | 1847.95 | 1852.34 | 1848.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 1848.00 | 1851.47 | 1848.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 13:15:00 | 1852.75 | 1851.53 | 1848.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 13:45:00 | 1852.90 | 1852.22 | 1849.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 09:15:00 | 1870.25 | 1854.78 | 1850.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 11:15:00 | 1854.45 | 1868.26 | 1863.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 1854.00 | 1865.41 | 1862.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 1847.95 | 1865.41 | 1862.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-01-06 12:15:00 | 1834.60 | 1859.25 | 1860.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2025-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 12:15:00 | 1834.60 | 1859.25 | 1860.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 1826.10 | 1852.62 | 1857.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 10:15:00 | 1843.90 | 1842.56 | 1849.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 10:15:00 | 1843.90 | 1842.56 | 1849.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 1843.90 | 1842.56 | 1849.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:30:00 | 1849.70 | 1842.56 | 1849.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 1853.90 | 1844.83 | 1850.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 12:00:00 | 1853.90 | 1844.83 | 1850.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 1876.55 | 1851.17 | 1852.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:00:00 | 1876.55 | 1851.17 | 1852.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2025-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 13:15:00 | 1892.65 | 1859.47 | 1856.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-07 14:15:00 | 1897.05 | 1866.98 | 1859.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 09:15:00 | 1884.50 | 1884.82 | 1875.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-09 09:45:00 | 1881.00 | 1884.82 | 1875.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 13:15:00 | 1879.15 | 1888.41 | 1880.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 14:00:00 | 1879.15 | 1888.41 | 1880.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 14:15:00 | 1873.05 | 1885.34 | 1880.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 14:45:00 | 1872.05 | 1885.34 | 1880.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 09:15:00 | 1845.10 | 1873.32 | 1875.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 11:15:00 | 1802.70 | 1853.19 | 1865.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 15:15:00 | 1835.00 | 1833.87 | 1850.74 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 09:15:00 | 1807.70 | 1833.87 | 1850.74 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 09:45:00 | 1815.25 | 1831.85 | 1848.29 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 10:30:00 | 1815.00 | 1828.22 | 1845.14 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 12:15:00 | 1812.30 | 1825.68 | 1842.45 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 1783.40 | 1770.98 | 1781.08 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-16 09:15:00 | 1783.40 | 1770.98 | 1781.08 | SL hit (close>ema400) qty=1.00 sl=1781.08 alert=retest1 |

### Cycle 48 — BUY (started 2025-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 13:15:00 | 1796.05 | 1773.49 | 1773.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 14:15:00 | 1806.20 | 1780.03 | 1776.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 14:15:00 | 1799.85 | 1800.78 | 1791.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 15:00:00 | 1799.85 | 1800.78 | 1791.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 1790.00 | 1798.62 | 1791.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 1783.35 | 1798.62 | 1791.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 1781.70 | 1795.24 | 1790.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 14:00:00 | 1805.00 | 1794.00 | 1790.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 14:30:00 | 1802.40 | 1796.17 | 1792.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 10:15:00 | 1803.45 | 1797.33 | 1793.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 11:45:00 | 1805.75 | 1800.49 | 1795.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 15:15:00 | 1809.00 | 1804.66 | 1799.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:15:00 | 1791.90 | 1804.66 | 1799.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 1791.35 | 1802.00 | 1798.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:45:00 | 1786.00 | 1802.00 | 1798.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 1797.10 | 1801.02 | 1798.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:45:00 | 1793.00 | 1801.02 | 1798.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 11:15:00 | 1799.50 | 1800.72 | 1798.59 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-24 14:15:00 | 1780.15 | 1795.27 | 1796.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2025-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 14:15:00 | 1780.15 | 1795.27 | 1796.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 1752.90 | 1785.93 | 1792.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 13:15:00 | 1727.00 | 1719.21 | 1733.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 14:00:00 | 1727.00 | 1719.21 | 1733.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 1756.95 | 1727.48 | 1734.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:45:00 | 1753.25 | 1727.48 | 1734.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 1757.05 | 1733.40 | 1736.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 11:15:00 | 1745.75 | 1733.40 | 1736.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 12:15:00 | 1755.80 | 1740.36 | 1739.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2025-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 12:15:00 | 1755.80 | 1740.36 | 1739.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 14:15:00 | 1767.20 | 1746.29 | 1743.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 10:15:00 | 1751.30 | 1752.20 | 1747.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 10:15:00 | 1751.30 | 1752.20 | 1747.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 1751.30 | 1752.20 | 1747.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:00:00 | 1751.30 | 1752.20 | 1747.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 1762.10 | 1754.18 | 1748.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 1753.25 | 1754.18 | 1748.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1747.50 | 1752.84 | 1748.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 1747.50 | 1752.84 | 1748.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1778.85 | 1758.05 | 1751.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 15:00:00 | 1789.90 | 1764.42 | 1754.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 11:15:00 | 1729.55 | 1750.89 | 1750.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 11:15:00 | 1729.55 | 1750.89 | 1750.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 13:15:00 | 1728.35 | 1743.29 | 1747.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 11:15:00 | 1763.15 | 1738.53 | 1742.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 11:15:00 | 1763.15 | 1738.53 | 1742.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 11:15:00 | 1763.15 | 1738.53 | 1742.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 12:00:00 | 1763.15 | 1738.53 | 1742.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 12:15:00 | 1731.15 | 1737.05 | 1741.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 09:15:00 | 1689.00 | 1718.06 | 1724.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 13:15:00 | 1604.55 | 1647.83 | 1670.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-12 10:15:00 | 1520.10 | 1601.17 | 1638.93 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 52 — BUY (started 2025-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 12:15:00 | 1689.45 | 1644.78 | 1639.51 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2025-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 11:15:00 | 1645.50 | 1654.34 | 1654.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-17 13:15:00 | 1638.25 | 1650.74 | 1652.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 15:15:00 | 1650.00 | 1648.98 | 1651.69 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-18 09:30:00 | 1624.35 | 1646.41 | 1650.28 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-18 10:15:00 | 1626.65 | 1646.41 | 1650.28 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 11:15:00 | 1626.30 | 1622.08 | 1632.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 12:00:00 | 1626.30 | 1622.08 | 1632.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 12:15:00 | 1667.75 | 1631.21 | 1635.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-19 12:15:00 | 1667.75 | 1631.21 | 1635.30 | SL hit (close>ema400) qty=1.00 sl=1635.30 alert=retest1 |

### Cycle 54 — BUY (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 13:15:00 | 1668.65 | 1638.70 | 1638.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 1674.25 | 1645.81 | 1641.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 12:15:00 | 1634.60 | 1656.70 | 1650.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-20 12:15:00 | 1634.60 | 1656.70 | 1650.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 12:15:00 | 1634.60 | 1656.70 | 1650.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 13:00:00 | 1634.60 | 1656.70 | 1650.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 13:15:00 | 1639.00 | 1653.16 | 1649.18 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2025-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 15:15:00 | 1625.00 | 1643.88 | 1645.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 09:15:00 | 1614.85 | 1638.07 | 1642.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 14:15:00 | 1617.95 | 1572.29 | 1585.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 14:15:00 | 1617.95 | 1572.29 | 1585.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 14:15:00 | 1617.95 | 1572.29 | 1585.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 15:00:00 | 1617.95 | 1572.29 | 1585.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 15:15:00 | 1577.50 | 1573.33 | 1584.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 09:15:00 | 1544.85 | 1573.33 | 1584.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 15:15:00 | 1552.10 | 1568.38 | 1575.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 13:15:00 | 1573.45 | 1563.04 | 1568.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 14:00:00 | 1572.15 | 1564.86 | 1569.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-28 14:15:00 | 1653.80 | 1582.65 | 1576.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — BUY (started 2025-02-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-28 14:15:00 | 1653.80 | 1582.65 | 1576.84 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2025-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-06 14:15:00 | 1592.90 | 1610.75 | 1611.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 14:15:00 | 1576.95 | 1597.88 | 1604.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-10 10:15:00 | 1598.65 | 1594.56 | 1600.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-10 11:00:00 | 1598.65 | 1594.56 | 1600.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 1592.25 | 1594.10 | 1599.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:30:00 | 1601.80 | 1594.10 | 1599.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 1599.05 | 1595.09 | 1599.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-10 13:00:00 | 1599.05 | 1595.09 | 1599.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 1590.55 | 1594.18 | 1598.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-11 09:15:00 | 1546.85 | 1592.68 | 1597.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-13 09:15:00 | 1469.51 | 1525.38 | 1549.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-13 10:15:00 | 1526.80 | 1525.66 | 1547.48 | SL hit (close>ema200) qty=0.50 sl=1525.66 alert=retest2 |

### Cycle 58 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 11:15:00 | 1557.65 | 1535.59 | 1534.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 13:15:00 | 1565.30 | 1543.50 | 1538.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 13:15:00 | 1630.35 | 1632.25 | 1613.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 14:00:00 | 1630.35 | 1632.25 | 1613.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 59 — SELL (started 2025-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 09:15:00 | 1594.00 | 1681.50 | 1681.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 09:15:00 | 1560.80 | 1617.56 | 1635.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 09:15:00 | 1567.20 | 1563.84 | 1592.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 10:00:00 | 1567.20 | 1563.84 | 1592.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 1575.80 | 1566.85 | 1579.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:30:00 | 1582.50 | 1566.85 | 1579.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 1577.40 | 1568.96 | 1579.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 11:00:00 | 1577.40 | 1568.96 | 1579.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 11:15:00 | 1578.05 | 1570.78 | 1579.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 11:30:00 | 1581.50 | 1570.78 | 1579.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 12:15:00 | 1582.40 | 1573.10 | 1579.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 12:45:00 | 1580.20 | 1573.10 | 1579.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 13:15:00 | 1581.55 | 1574.79 | 1579.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 14:00:00 | 1581.55 | 1574.79 | 1579.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1521.70 | 1565.82 | 1574.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 10:15:00 | 1514.90 | 1565.82 | 1574.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 11:30:00 | 1516.70 | 1548.20 | 1564.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 09:15:00 | 1445.50 | 1539.94 | 1554.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 1439.15 | 1522.71 | 1545.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 1440.87 | 1522.71 | 1545.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-07 13:15:00 | 1499.50 | 1498.68 | 1525.02 | SL hit (close>ema200) qty=0.50 sl=1498.68 alert=retest2 |

### Cycle 60 — BUY (started 2025-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 12:15:00 | 1531.80 | 1524.27 | 1523.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 13:15:00 | 1540.45 | 1527.51 | 1525.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 11:15:00 | 1613.20 | 1618.40 | 1601.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 11:15:00 | 1613.20 | 1618.40 | 1601.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 11:15:00 | 1613.20 | 1618.40 | 1601.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 12:00:00 | 1613.20 | 1618.40 | 1601.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 1655.90 | 1658.87 | 1649.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 09:15:00 | 1637.00 | 1658.87 | 1649.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 1603.80 | 1647.86 | 1645.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 1603.80 | 1647.86 | 1645.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2025-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 10:15:00 | 1598.90 | 1638.07 | 1640.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-23 13:15:00 | 1584.70 | 1613.99 | 1627.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 12:15:00 | 1597.70 | 1596.91 | 1611.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-24 12:45:00 | 1602.00 | 1596.91 | 1611.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 1605.70 | 1598.67 | 1610.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 14:00:00 | 1605.70 | 1598.67 | 1610.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 15:15:00 | 1612.90 | 1603.25 | 1610.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-25 09:15:00 | 1605.80 | 1603.25 | 1610.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 1595.20 | 1601.64 | 1609.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 14:00:00 | 1567.50 | 1587.71 | 1599.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 14:45:00 | 1570.00 | 1585.71 | 1597.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 15:15:00 | 1570.00 | 1585.71 | 1597.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-30 10:15:00 | 1602.40 | 1594.79 | 1594.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 10:15:00 | 1602.40 | 1594.79 | 1594.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 13:15:00 | 1614.20 | 1601.76 | 1598.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 09:15:00 | 1601.60 | 1605.83 | 1601.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 1601.60 | 1605.83 | 1601.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 1601.60 | 1605.83 | 1601.19 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2025-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 11:15:00 | 1578.20 | 1598.14 | 1598.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 12:15:00 | 1574.10 | 1593.33 | 1596.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 1530.90 | 1528.63 | 1548.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 10:30:00 | 1527.10 | 1528.63 | 1548.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 1540.00 | 1531.53 | 1542.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 1548.40 | 1531.53 | 1542.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1556.30 | 1536.48 | 1543.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:30:00 | 1559.60 | 1536.48 | 1543.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 1554.20 | 1540.02 | 1544.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:30:00 | 1547.40 | 1544.87 | 1545.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 11:15:00 | 1549.40 | 1538.26 | 1538.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1549.40 | 1538.26 | 1538.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 1553.00 | 1541.21 | 1539.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 1663.80 | 1668.43 | 1647.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 13:30:00 | 1663.40 | 1668.43 | 1647.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1653.10 | 1663.87 | 1650.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 1651.50 | 1663.87 | 1650.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 1652.10 | 1659.65 | 1650.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 1652.80 | 1659.65 | 1650.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 1663.90 | 1660.50 | 1651.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:30:00 | 1652.40 | 1660.50 | 1651.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 13:15:00 | 1668.20 | 1675.32 | 1667.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 14:00:00 | 1668.20 | 1675.32 | 1667.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 1673.40 | 1674.93 | 1667.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 15:15:00 | 1667.20 | 1674.93 | 1667.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 15:15:00 | 1667.20 | 1673.39 | 1667.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:15:00 | 1661.40 | 1673.39 | 1667.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 1652.10 | 1669.13 | 1666.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:45:00 | 1655.90 | 1669.13 | 1666.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 1666.20 | 1668.54 | 1666.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 11:45:00 | 1689.00 | 1670.16 | 1667.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 12:30:00 | 1676.50 | 1671.30 | 1667.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 1669.50 | 1691.45 | 1693.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 09:15:00 | 1669.50 | 1691.45 | 1693.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 10:15:00 | 1665.30 | 1686.22 | 1690.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 1677.50 | 1676.57 | 1683.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 1677.50 | 1676.57 | 1683.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 1677.50 | 1676.57 | 1683.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:00:00 | 1677.50 | 1676.57 | 1683.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 1690.50 | 1679.39 | 1683.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:30:00 | 1693.80 | 1679.39 | 1683.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 1685.40 | 1680.59 | 1683.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 13:45:00 | 1677.00 | 1680.87 | 1683.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 14:15:00 | 1693.40 | 1683.38 | 1684.20 | SL hit (close>static) qty=1.00 sl=1690.50 alert=retest2 |

### Cycle 66 — BUY (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 15:15:00 | 1695.50 | 1685.80 | 1685.23 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 1678.60 | 1684.36 | 1684.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 11:15:00 | 1666.80 | 1680.66 | 1682.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 15:15:00 | 1684.00 | 1677.34 | 1680.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 15:15:00 | 1684.00 | 1677.34 | 1680.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 1684.00 | 1677.34 | 1680.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 1681.00 | 1677.34 | 1680.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 1688.80 | 1679.63 | 1680.98 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 11:15:00 | 1683.90 | 1681.84 | 1681.83 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2025-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 14:15:00 | 1674.70 | 1681.02 | 1681.52 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 1698.90 | 1684.44 | 1682.98 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 11:15:00 | 1687.20 | 1698.58 | 1698.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-09 13:15:00 | 1681.50 | 1693.62 | 1696.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 09:15:00 | 1689.70 | 1677.58 | 1683.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 1689.70 | 1677.58 | 1683.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1689.70 | 1677.58 | 1683.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:15:00 | 1702.50 | 1677.58 | 1683.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 1698.70 | 1681.80 | 1684.59 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2025-06-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 11:15:00 | 1709.90 | 1687.42 | 1686.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 12:15:00 | 1736.70 | 1697.28 | 1691.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 1725.00 | 1731.71 | 1716.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 14:00:00 | 1725.00 | 1731.71 | 1716.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 1721.10 | 1731.93 | 1720.35 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2025-06-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 13:15:00 | 1697.30 | 1714.50 | 1715.00 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2025-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 14:15:00 | 1735.00 | 1718.60 | 1716.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 14:15:00 | 1738.00 | 1727.41 | 1722.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 14:15:00 | 1728.30 | 1733.64 | 1728.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 14:15:00 | 1728.30 | 1733.64 | 1728.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 1728.30 | 1733.64 | 1728.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 14:30:00 | 1727.00 | 1733.64 | 1728.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 1719.00 | 1730.71 | 1728.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:15:00 | 1719.30 | 1730.71 | 1728.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 1723.50 | 1729.27 | 1727.65 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 10:15:00 | 1713.50 | 1726.12 | 1726.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 1707.50 | 1719.19 | 1722.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 1704.70 | 1691.56 | 1703.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 1704.70 | 1691.56 | 1703.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1704.70 | 1691.56 | 1703.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:00:00 | 1704.70 | 1691.56 | 1703.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1695.70 | 1692.39 | 1702.96 | EMA400 retest candle locked (from downside) |

### Cycle 76 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 1772.70 | 1712.52 | 1709.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 12:15:00 | 1812.40 | 1757.13 | 1734.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 13:15:00 | 1773.40 | 1791.83 | 1780.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 13:15:00 | 1773.40 | 1791.83 | 1780.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 1773.40 | 1791.83 | 1780.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:00:00 | 1773.40 | 1791.83 | 1780.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 1798.70 | 1793.20 | 1781.89 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2025-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 13:15:00 | 1754.50 | 1774.39 | 1776.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 14:15:00 | 1751.00 | 1769.72 | 1774.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 10:15:00 | 1769.00 | 1765.40 | 1770.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 10:15:00 | 1769.00 | 1765.40 | 1770.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 1769.00 | 1765.40 | 1770.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 11:00:00 | 1769.00 | 1765.40 | 1770.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 1772.00 | 1766.72 | 1771.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 11:45:00 | 1773.00 | 1766.72 | 1771.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 1782.70 | 1769.92 | 1772.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 12:30:00 | 1784.50 | 1769.92 | 1772.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 1781.30 | 1772.20 | 1772.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 14:15:00 | 1782.00 | 1772.20 | 1772.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2025-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 14:15:00 | 1800.00 | 1777.76 | 1775.40 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2025-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 09:15:00 | 1685.00 | 1761.52 | 1768.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 09:15:00 | 1642.10 | 1689.40 | 1722.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 14:15:00 | 1641.00 | 1640.85 | 1664.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-02 14:45:00 | 1641.80 | 1640.85 | 1664.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1628.50 | 1626.43 | 1631.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 13:15:00 | 1615.00 | 1622.48 | 1626.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 14:00:00 | 1614.70 | 1620.93 | 1625.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 14:45:00 | 1614.60 | 1619.44 | 1624.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:15:00 | 1614.40 | 1618.04 | 1622.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 1616.80 | 1612.19 | 1616.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 1619.20 | 1612.19 | 1616.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 1623.50 | 1614.45 | 1617.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:00:00 | 1623.50 | 1614.45 | 1617.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 1633.90 | 1618.34 | 1618.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:00:00 | 1633.90 | 1618.34 | 1618.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-14 12:15:00 | 1634.00 | 1621.47 | 1620.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 12:15:00 | 1634.00 | 1621.47 | 1620.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 1642.90 | 1631.48 | 1625.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 12:15:00 | 1645.50 | 1650.27 | 1642.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 13:00:00 | 1645.50 | 1650.27 | 1642.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 1649.70 | 1650.16 | 1642.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 14:45:00 | 1653.90 | 1651.21 | 1643.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 13:00:00 | 1660.00 | 1657.03 | 1653.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 09:45:00 | 1653.10 | 1654.98 | 1653.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 10:15:00 | 1748.10 | 1763.19 | 1763.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2025-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 10:15:00 | 1748.10 | 1763.19 | 1763.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 1736.00 | 1757.76 | 1761.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 11:15:00 | 1672.00 | 1670.47 | 1685.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 12:00:00 | 1672.00 | 1670.47 | 1685.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 1689.20 | 1677.80 | 1685.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:45:00 | 1689.80 | 1677.80 | 1685.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 1690.00 | 1680.24 | 1685.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 1682.00 | 1680.24 | 1685.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 1691.60 | 1683.61 | 1686.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:00:00 | 1691.60 | 1683.61 | 1686.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 1690.40 | 1684.97 | 1686.93 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2025-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 14:15:00 | 1690.60 | 1688.34 | 1688.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 09:15:00 | 1695.40 | 1690.34 | 1689.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 09:15:00 | 1701.10 | 1702.98 | 1697.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 09:15:00 | 1701.10 | 1702.98 | 1697.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 1701.10 | 1702.98 | 1697.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:30:00 | 1698.40 | 1702.98 | 1697.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 1695.20 | 1700.81 | 1697.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 12:00:00 | 1695.20 | 1700.81 | 1697.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 1707.20 | 1702.08 | 1698.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 12:30:00 | 1702.00 | 1702.08 | 1698.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 1719.10 | 1711.03 | 1704.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:45:00 | 1708.40 | 1711.03 | 1704.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 1707.10 | 1713.56 | 1708.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 1707.10 | 1713.56 | 1708.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 1721.20 | 1715.09 | 1709.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 11:45:00 | 1724.70 | 1717.43 | 1711.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 12:45:00 | 1724.70 | 1717.48 | 1712.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 12:00:00 | 1724.50 | 1717.37 | 1715.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 13:15:00 | 1727.60 | 1718.11 | 1715.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1729.10 | 1728.14 | 1724.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 1715.50 | 1728.14 | 1724.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 1728.00 | 1728.45 | 1725.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:30:00 | 1724.70 | 1728.45 | 1725.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 1729.00 | 1728.56 | 1726.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:45:00 | 1727.20 | 1728.56 | 1726.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 1721.10 | 1727.07 | 1725.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 1725.80 | 1727.07 | 1725.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1733.10 | 1728.27 | 1726.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 13:15:00 | 1739.80 | 1731.62 | 1728.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 09:45:00 | 1739.00 | 1747.06 | 1742.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 11:15:00 | 1739.50 | 1745.33 | 1741.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 13:15:00 | 1730.10 | 1739.01 | 1739.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 13:15:00 | 1730.10 | 1739.01 | 1739.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 1717.70 | 1734.75 | 1737.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1728.60 | 1713.82 | 1720.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 1728.60 | 1713.82 | 1720.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 1728.60 | 1713.82 | 1720.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 1728.60 | 1713.82 | 1720.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 1725.20 | 1716.09 | 1720.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 1727.70 | 1716.09 | 1720.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 1717.00 | 1716.87 | 1720.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 13:30:00 | 1722.90 | 1716.87 | 1720.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 1722.40 | 1717.97 | 1720.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 15:00:00 | 1722.40 | 1717.97 | 1720.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 1721.40 | 1718.66 | 1720.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:15:00 | 1724.20 | 1718.66 | 1720.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 1730.20 | 1722.13 | 1721.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 13:15:00 | 1735.00 | 1726.51 | 1724.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-01 15:15:00 | 1713.70 | 1724.37 | 1723.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 15:15:00 | 1713.70 | 1724.37 | 1723.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 1713.70 | 1724.37 | 1723.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 1704.70 | 1724.37 | 1723.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1721.90 | 1723.87 | 1723.36 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 12:15:00 | 1716.80 | 1722.48 | 1722.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 15:15:00 | 1711.30 | 1718.74 | 1720.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 10:15:00 | 1723.30 | 1719.61 | 1720.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 10:15:00 | 1723.30 | 1719.61 | 1720.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 1723.30 | 1719.61 | 1720.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:00:00 | 1723.30 | 1719.61 | 1720.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 11:15:00 | 1731.90 | 1722.07 | 1721.94 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 1719.00 | 1723.18 | 1723.31 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 09:15:00 | 1732.50 | 1724.44 | 1723.82 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 12:15:00 | 1710.00 | 1721.32 | 1722.61 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 11:15:00 | 1730.10 | 1723.13 | 1722.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 1740.00 | 1728.78 | 1725.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 10:15:00 | 1727.00 | 1728.42 | 1726.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 10:15:00 | 1727.00 | 1728.42 | 1726.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 1727.00 | 1728.42 | 1726.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:45:00 | 1731.00 | 1728.42 | 1726.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 1711.30 | 1725.00 | 1724.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:00:00 | 1711.30 | 1725.00 | 1724.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 12:15:00 | 1705.20 | 1721.04 | 1722.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 09:15:00 | 1700.80 | 1712.75 | 1717.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 11:15:00 | 1687.60 | 1684.43 | 1691.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 11:30:00 | 1688.40 | 1684.43 | 1691.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 1690.00 | 1685.87 | 1689.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:15:00 | 1687.70 | 1685.87 | 1689.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 1683.70 | 1685.44 | 1689.27 | EMA400 retest candle locked (from downside) |

### Cycle 92 — BUY (started 2025-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 15:15:00 | 1694.80 | 1690.06 | 1689.95 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 09:15:00 | 1685.40 | 1689.13 | 1689.54 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 09:15:00 | 1694.00 | 1690.36 | 1689.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 10:15:00 | 1705.60 | 1693.41 | 1691.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 15:15:00 | 1720.00 | 1724.11 | 1717.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-24 09:15:00 | 1715.30 | 1724.11 | 1717.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1706.00 | 1720.49 | 1716.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 1706.00 | 1720.49 | 1716.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 1702.00 | 1716.79 | 1714.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:30:00 | 1704.30 | 1716.79 | 1714.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 11:15:00 | 1699.90 | 1713.41 | 1713.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 1694.10 | 1706.19 | 1709.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 12:15:00 | 1670.80 | 1670.29 | 1683.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-26 12:30:00 | 1668.50 | 1670.29 | 1683.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 1672.30 | 1670.69 | 1682.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 13:45:00 | 1677.20 | 1670.69 | 1682.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1695.30 | 1675.88 | 1681.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:00:00 | 1695.30 | 1675.88 | 1681.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 1698.30 | 1680.37 | 1683.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:00:00 | 1698.30 | 1680.37 | 1683.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2025-09-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 12:15:00 | 1698.10 | 1686.25 | 1685.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 14:15:00 | 1707.00 | 1692.73 | 1688.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-30 09:15:00 | 1687.00 | 1694.92 | 1690.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 09:15:00 | 1687.00 | 1694.92 | 1690.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 1687.00 | 1694.92 | 1690.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:00:00 | 1687.00 | 1694.92 | 1690.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 1684.90 | 1692.92 | 1690.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:30:00 | 1682.00 | 1692.92 | 1690.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 1686.40 | 1697.32 | 1694.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:00:00 | 1686.40 | 1697.32 | 1694.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 1694.70 | 1696.79 | 1694.82 | EMA400 retest candle locked (from upside) |

### Cycle 97 — SELL (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 15:15:00 | 1687.00 | 1693.62 | 1693.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-03 09:15:00 | 1676.60 | 1690.22 | 1692.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 14:15:00 | 1659.00 | 1655.51 | 1662.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-07 15:00:00 | 1659.00 | 1655.51 | 1662.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 1658.00 | 1656.01 | 1662.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:15:00 | 1668.70 | 1656.01 | 1662.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1660.10 | 1656.83 | 1661.99 | EMA400 retest candle locked (from downside) |

### Cycle 98 — BUY (started 2025-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 13:15:00 | 1673.30 | 1664.26 | 1664.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-08 15:15:00 | 1678.70 | 1669.12 | 1666.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-09 10:15:00 | 1667.00 | 1669.19 | 1667.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 10:15:00 | 1667.00 | 1669.19 | 1667.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 1667.00 | 1669.19 | 1667.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:45:00 | 1664.70 | 1669.19 | 1667.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 1666.70 | 1668.69 | 1667.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 1666.70 | 1668.69 | 1667.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 1675.40 | 1670.03 | 1667.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:30:00 | 1667.00 | 1670.03 | 1667.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1666.50 | 1670.01 | 1668.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 11:30:00 | 1679.90 | 1674.00 | 1670.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 13:30:00 | 1678.40 | 1674.71 | 1671.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 13:15:00 | 1666.70 | 1670.12 | 1670.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 1666.70 | 1670.12 | 1670.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 14:15:00 | 1663.10 | 1668.72 | 1669.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 14:15:00 | 1659.80 | 1659.34 | 1663.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-14 15:00:00 | 1659.80 | 1659.34 | 1663.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1675.30 | 1662.00 | 1663.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 1675.30 | 1662.00 | 1663.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1677.80 | 1665.16 | 1665.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:45:00 | 1679.20 | 1665.16 | 1665.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 1679.40 | 1668.01 | 1666.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 10:15:00 | 1685.00 | 1674.40 | 1671.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 15:15:00 | 1685.00 | 1685.48 | 1678.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 09:15:00 | 1691.10 | 1685.48 | 1678.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 1695.30 | 1687.44 | 1680.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 09:15:00 | 1709.50 | 1692.96 | 1688.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 11:00:00 | 1712.90 | 1698.92 | 1691.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 14:00:00 | 1712.00 | 1705.67 | 1697.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 15:15:00 | 1710.20 | 1705.16 | 1697.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 1705.80 | 1706.09 | 1699.44 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-28 11:15:00 | 1690.00 | 1697.99 | 1698.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 11:15:00 | 1690.00 | 1697.99 | 1698.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 12:15:00 | 1685.80 | 1695.55 | 1697.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 1692.00 | 1690.65 | 1694.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 1692.00 | 1690.65 | 1694.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1692.00 | 1690.65 | 1694.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:15:00 | 1697.50 | 1690.65 | 1694.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 1702.80 | 1693.08 | 1695.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 1702.80 | 1693.08 | 1695.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 1701.50 | 1694.77 | 1695.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:30:00 | 1702.90 | 1694.77 | 1695.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 1693.80 | 1694.51 | 1695.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:30:00 | 1700.00 | 1694.51 | 1695.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 1696.20 | 1694.85 | 1695.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:15:00 | 1692.20 | 1694.85 | 1695.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 11:15:00 | 1699.20 | 1696.23 | 1695.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 11:15:00 | 1699.20 | 1696.23 | 1695.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 14:15:00 | 1705.40 | 1699.29 | 1697.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 09:15:00 | 1695.00 | 1699.98 | 1698.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 09:15:00 | 1695.00 | 1699.98 | 1698.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1695.00 | 1699.98 | 1698.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 1695.00 | 1699.98 | 1698.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1696.00 | 1699.19 | 1698.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:30:00 | 1689.10 | 1699.19 | 1698.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 1695.60 | 1697.61 | 1697.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:30:00 | 1691.70 | 1697.61 | 1697.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 13:15:00 | 1688.50 | 1695.79 | 1696.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 1679.50 | 1692.53 | 1695.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 1699.30 | 1691.61 | 1694.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 1699.30 | 1691.61 | 1694.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1699.30 | 1691.61 | 1694.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 1699.30 | 1691.61 | 1694.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 1701.80 | 1693.64 | 1694.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:45:00 | 1700.20 | 1693.64 | 1694.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 11:15:00 | 1703.80 | 1695.68 | 1695.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 12:15:00 | 1709.00 | 1698.34 | 1696.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 11:15:00 | 1707.40 | 1708.53 | 1703.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 11:30:00 | 1705.80 | 1708.53 | 1703.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 1707.70 | 1708.36 | 1703.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 12:45:00 | 1708.00 | 1708.36 | 1703.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 1704.80 | 1707.65 | 1704.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 13:45:00 | 1706.90 | 1707.65 | 1704.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 1697.00 | 1705.52 | 1703.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 15:00:00 | 1697.00 | 1705.52 | 1703.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 1699.40 | 1704.30 | 1703.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:15:00 | 1691.50 | 1704.30 | 1703.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 1690.70 | 1701.58 | 1701.91 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2025-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 14:15:00 | 1704.50 | 1699.06 | 1698.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 09:15:00 | 1803.20 | 1720.58 | 1708.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 09:15:00 | 1815.90 | 1819.39 | 1792.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 09:45:00 | 1812.50 | 1819.39 | 1792.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1830.40 | 1822.79 | 1807.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 10:15:00 | 1837.00 | 1822.79 | 1807.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 09:30:00 | 1832.90 | 1825.17 | 1816.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 10:00:00 | 1833.70 | 1825.17 | 1816.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 09:15:00 | 1799.50 | 1816.64 | 1817.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 1799.50 | 1816.64 | 1817.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 10:15:00 | 1795.10 | 1812.33 | 1815.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 1743.60 | 1737.67 | 1751.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 10:00:00 | 1743.60 | 1737.67 | 1751.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 1749.00 | 1739.94 | 1750.92 | EMA400 retest candle locked (from downside) |

### Cycle 108 — BUY (started 2025-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 11:15:00 | 1765.40 | 1755.28 | 1754.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 14:15:00 | 1770.80 | 1760.79 | 1757.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 15:15:00 | 1776.60 | 1778.00 | 1770.19 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 09:45:00 | 1786.80 | 1779.20 | 1771.45 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 10:15:00 | 1787.10 | 1779.20 | 1771.45 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 1772.00 | 1777.09 | 1771.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:00:00 | 1772.00 | 1777.09 | 1771.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 1772.10 | 1776.09 | 1771.82 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-27 13:15:00 | 1770.00 | 1774.87 | 1771.65 | SL hit (close<ema400) qty=1.00 sl=1771.65 alert=retest1 |

### Cycle 109 — SELL (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 09:15:00 | 1765.70 | 1770.34 | 1770.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 09:15:00 | 1757.30 | 1767.35 | 1769.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 13:15:00 | 1763.10 | 1761.26 | 1765.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-02 13:45:00 | 1760.60 | 1761.26 | 1765.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 1763.00 | 1762.05 | 1764.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 1776.30 | 1762.05 | 1764.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1769.30 | 1763.50 | 1765.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:45:00 | 1761.20 | 1763.36 | 1765.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 13:15:00 | 1771.80 | 1766.63 | 1766.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2025-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 13:15:00 | 1771.80 | 1766.63 | 1766.27 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2025-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 15:15:00 | 1763.20 | 1765.70 | 1765.90 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 09:15:00 | 1779.50 | 1768.46 | 1767.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 10:15:00 | 1803.90 | 1775.55 | 1770.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 1817.10 | 1821.63 | 1808.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 10:00:00 | 1817.10 | 1821.63 | 1808.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 1809.40 | 1819.13 | 1810.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:45:00 | 1809.90 | 1819.13 | 1810.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 1803.60 | 1816.03 | 1809.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:00:00 | 1803.60 | 1816.03 | 1809.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 1801.30 | 1813.08 | 1809.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:45:00 | 1799.80 | 1813.08 | 1809.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 1792.90 | 1808.68 | 1807.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:45:00 | 1792.50 | 1808.68 | 1807.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 10:15:00 | 1792.70 | 1805.48 | 1806.35 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 13:15:00 | 1813.70 | 1804.40 | 1803.89 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 13:15:00 | 1799.00 | 1806.73 | 1807.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 09:15:00 | 1785.00 | 1799.92 | 1803.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 14:15:00 | 1793.10 | 1792.61 | 1798.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 14:15:00 | 1793.10 | 1792.61 | 1798.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 1793.10 | 1792.61 | 1798.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 15:15:00 | 1796.00 | 1792.61 | 1798.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 1796.00 | 1793.28 | 1798.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 1787.70 | 1793.28 | 1798.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 10:00:00 | 1787.10 | 1792.05 | 1797.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 10:45:00 | 1790.60 | 1792.06 | 1796.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 11:15:00 | 1787.10 | 1792.06 | 1796.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 1782.40 | 1781.11 | 1786.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 13:30:00 | 1781.60 | 1781.11 | 1786.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 1797.90 | 1784.47 | 1787.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:00:00 | 1797.90 | 1784.47 | 1787.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 1789.90 | 1785.55 | 1787.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 1783.30 | 1785.55 | 1787.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 09:45:00 | 1789.40 | 1785.34 | 1785.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:00:00 | 1788.60 | 1785.99 | 1786.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 12:15:00 | 1789.90 | 1786.26 | 1786.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — BUY (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 12:15:00 | 1789.90 | 1786.26 | 1786.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 1796.50 | 1788.31 | 1787.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 10:15:00 | 1802.70 | 1803.93 | 1799.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 10:15:00 | 1802.70 | 1803.93 | 1799.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 1802.70 | 1803.93 | 1799.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:15:00 | 1805.80 | 1802.89 | 1800.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 11:45:00 | 1807.10 | 1804.19 | 1801.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:15:00 | 1805.30 | 1810.00 | 1806.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 10:00:00 | 1808.40 | 1813.33 | 1810.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 1802.60 | 1811.19 | 1809.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:00:00 | 1802.60 | 1811.19 | 1809.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-29 12:15:00 | 1802.30 | 1808.27 | 1808.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 1802.30 | 1808.27 | 1808.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 13:15:00 | 1791.00 | 1804.82 | 1806.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 12:15:00 | 1797.70 | 1797.02 | 1801.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 12:15:00 | 1797.70 | 1797.02 | 1801.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 1797.70 | 1797.02 | 1801.54 | EMA400 retest candle locked (from downside) |

### Cycle 118 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 1815.50 | 1803.65 | 1803.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 10:15:00 | 1822.00 | 1814.39 | 1809.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 12:15:00 | 1813.90 | 1814.80 | 1810.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 13:00:00 | 1813.90 | 1814.80 | 1810.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 1826.60 | 1819.09 | 1814.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 10:15:00 | 1829.60 | 1819.09 | 1814.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 09:15:00 | 1863.40 | 1888.58 | 1891.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2026-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 09:15:00 | 1863.40 | 1888.58 | 1891.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 13:15:00 | 1859.90 | 1865.76 | 1873.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 1869.00 | 1866.41 | 1873.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 15:00:00 | 1869.00 | 1866.41 | 1873.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 1873.00 | 1867.13 | 1872.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 1873.00 | 1867.13 | 1872.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 1878.00 | 1869.30 | 1872.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:45:00 | 1884.00 | 1869.30 | 1872.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 13:15:00 | 1889.20 | 1874.88 | 1874.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 14:15:00 | 1894.80 | 1878.87 | 1876.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 1905.10 | 1913.56 | 1904.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 09:15:00 | 1905.10 | 1913.56 | 1904.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 1905.10 | 1913.56 | 1904.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:00:00 | 1905.10 | 1913.56 | 1904.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 1900.30 | 1910.91 | 1903.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:00:00 | 1900.30 | 1910.91 | 1903.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 1908.30 | 1910.39 | 1904.11 | EMA400 retest candle locked (from upside) |

### Cycle 121 — SELL (started 2026-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 14:15:00 | 1884.10 | 1900.95 | 1901.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 10:15:00 | 1863.90 | 1888.67 | 1894.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 1893.60 | 1888.11 | 1893.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 12:15:00 | 1893.60 | 1888.11 | 1893.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 1893.60 | 1888.11 | 1893.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 12:45:00 | 1890.10 | 1888.11 | 1893.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 1870.90 | 1884.67 | 1891.48 | EMA400 retest candle locked (from downside) |

### Cycle 122 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 1899.90 | 1893.83 | 1893.29 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 1886.20 | 1891.89 | 1892.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 1857.30 | 1884.97 | 1889.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 1871.60 | 1870.12 | 1880.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 1871.60 | 1870.12 | 1880.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 1871.60 | 1870.12 | 1880.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:30:00 | 1874.10 | 1870.12 | 1880.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 13:15:00 | 1868.70 | 1868.53 | 1876.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 13:30:00 | 1868.00 | 1868.53 | 1876.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 1879.00 | 1870.62 | 1876.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 15:00:00 | 1879.00 | 1870.62 | 1876.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 1882.50 | 1873.00 | 1876.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 1880.00 | 1873.00 | 1876.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1869.50 | 1872.30 | 1876.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 10:45:00 | 1852.70 | 1870.50 | 1875.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 11:15:00 | 1856.80 | 1870.50 | 1875.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 12:00:00 | 1854.90 | 1867.38 | 1873.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 13:15:00 | 1856.80 | 1865.50 | 1871.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1854.50 | 1850.49 | 1857.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:45:00 | 1862.10 | 1850.49 | 1857.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 1857.40 | 1851.87 | 1857.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:30:00 | 1866.90 | 1851.87 | 1857.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 1862.10 | 1853.92 | 1857.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:45:00 | 1862.80 | 1853.92 | 1857.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 1869.60 | 1857.05 | 1858.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:45:00 | 1873.00 | 1857.05 | 1858.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-30 13:15:00 | 1879.80 | 1861.60 | 1860.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 1879.80 | 1861.60 | 1860.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 1887.00 | 1868.78 | 1864.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 1852.10 | 1870.72 | 1868.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 1852.10 | 1870.72 | 1868.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 1852.10 | 1870.72 | 1868.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:45:00 | 1855.10 | 1870.72 | 1868.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 1835.90 | 1863.76 | 1865.49 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 1880.00 | 1864.74 | 1864.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 1898.80 | 1871.55 | 1867.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 1882.90 | 1887.57 | 1880.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 10:15:00 | 1882.90 | 1887.57 | 1880.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 1882.90 | 1887.57 | 1880.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:30:00 | 1878.60 | 1887.57 | 1880.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 1891.50 | 1888.35 | 1881.39 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 1863.50 | 1879.26 | 1880.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 1854.30 | 1863.35 | 1870.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 1869.20 | 1856.15 | 1861.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 1869.20 | 1856.15 | 1861.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 1869.20 | 1856.15 | 1861.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 1869.20 | 1856.15 | 1861.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 1893.60 | 1863.64 | 1864.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 1893.60 | 1863.64 | 1864.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 1884.80 | 1867.87 | 1866.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 1898.00 | 1882.30 | 1874.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 1899.40 | 1905.60 | 1892.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 10:00:00 | 1899.40 | 1905.60 | 1892.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 1894.00 | 1904.24 | 1895.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:45:00 | 1894.70 | 1904.24 | 1895.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 1892.70 | 1901.93 | 1895.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 14:00:00 | 1892.70 | 1901.93 | 1895.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 1894.60 | 1900.47 | 1895.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 14:30:00 | 1892.20 | 1900.47 | 1895.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 1894.90 | 1899.35 | 1895.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:15:00 | 1885.90 | 1899.35 | 1895.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 1897.10 | 1897.39 | 1894.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:00:00 | 1906.00 | 1899.11 | 1895.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 09:30:00 | 1916.90 | 1907.18 | 1901.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 15:15:00 | 1889.80 | 1900.89 | 1901.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 15:15:00 | 1889.80 | 1900.89 | 1901.11 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 09:15:00 | 2002.00 | 1921.11 | 1910.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 2031.90 | 2004.48 | 1993.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 12:15:00 | 2078.60 | 2084.50 | 2070.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 13:00:00 | 2078.60 | 2084.50 | 2070.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 2074.00 | 2081.08 | 2072.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:15:00 | 2073.20 | 2081.08 | 2072.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 2074.60 | 2079.78 | 2073.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:15:00 | 2066.00 | 2079.78 | 2073.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 2066.00 | 2077.03 | 2072.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:00:00 | 2066.00 | 2077.03 | 2072.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 2061.90 | 2074.00 | 2071.48 | EMA400 retest candle locked (from upside) |

### Cycle 131 — SELL (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 14:15:00 | 2056.00 | 2068.53 | 2069.32 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2026-03-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 14:15:00 | 2076.70 | 2067.41 | 2067.35 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 2035.90 | 2062.32 | 2065.13 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2026-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 12:15:00 | 2069.00 | 2059.83 | 2059.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 15:15:00 | 2074.70 | 2065.09 | 2062.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 09:15:00 | 2057.00 | 2063.47 | 2061.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 09:15:00 | 2057.00 | 2063.47 | 2061.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 2057.00 | 2063.47 | 2061.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:00:00 | 2057.00 | 2063.47 | 2061.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — SELL (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 10:15:00 | 2044.60 | 2059.70 | 2060.08 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 12:15:00 | 2062.70 | 2060.59 | 2060.44 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 13:15:00 | 2058.20 | 2060.11 | 2060.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 14:15:00 | 2051.50 | 2058.39 | 2059.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 12:15:00 | 2052.90 | 2045.77 | 2051.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 12:15:00 | 2052.90 | 2045.77 | 2051.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 12:15:00 | 2052.90 | 2045.77 | 2051.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 13:00:00 | 2052.90 | 2045.77 | 2051.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 13:15:00 | 2049.00 | 2046.42 | 2051.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 14:15:00 | 2057.00 | 2046.42 | 2051.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 14:15:00 | 2071.40 | 2051.41 | 2053.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 15:00:00 | 2071.40 | 2051.41 | 2053.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2026-03-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 15:15:00 | 2071.80 | 2055.49 | 2054.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 09:15:00 | 2102.70 | 2064.93 | 2059.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 2115.80 | 2119.58 | 2104.85 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 13:00:00 | 2135.00 | 2124.82 | 2111.09 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 2113.20 | 2125.72 | 2116.33 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 2113.20 | 2125.72 | 2116.33 | SL hit (close<ema400) qty=1.00 sl=2116.33 alert=retest1 |

### Cycle 139 — SELL (started 2026-03-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 11:15:00 | 2094.10 | 2111.55 | 2113.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 13:15:00 | 2070.00 | 2098.51 | 2106.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 2107.00 | 2097.74 | 2104.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 2107.00 | 2097.74 | 2104.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 2107.00 | 2097.74 | 2104.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 2107.00 | 2097.74 | 2104.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 2110.20 | 2100.23 | 2104.61 | EMA400 retest candle locked (from downside) |

### Cycle 140 — BUY (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 15:15:00 | 2115.00 | 2108.16 | 2107.39 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 09:15:00 | 2093.00 | 2105.12 | 2106.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-18 10:15:00 | 2082.00 | 2100.50 | 2103.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 12:15:00 | 2100.30 | 2099.58 | 2102.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-18 13:00:00 | 2100.30 | 2099.58 | 2102.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 142 — BUY (started 2026-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 14:15:00 | 2127.00 | 2104.44 | 2104.44 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 2096.60 | 2104.70 | 2104.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 2085.00 | 2099.76 | 2102.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 2106.20 | 2096.84 | 2100.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 2106.20 | 2096.84 | 2100.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 2106.20 | 2096.84 | 2100.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 2109.50 | 2096.84 | 2100.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 2105.50 | 2098.57 | 2100.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 2105.50 | 2098.57 | 2100.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 2110.00 | 2100.86 | 2101.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:45:00 | 2111.80 | 2100.86 | 2101.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — BUY (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 12:15:00 | 2116.80 | 2104.04 | 2102.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 13:15:00 | 2125.80 | 2108.40 | 2104.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 2093.60 | 2118.84 | 2111.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 09:15:00 | 2093.60 | 2118.84 | 2111.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 2093.60 | 2118.84 | 2111.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 2093.60 | 2118.84 | 2111.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 2082.00 | 2111.47 | 2108.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:30:00 | 2078.50 | 2111.47 | 2108.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — SELL (started 2026-03-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 11:15:00 | 2084.40 | 2106.06 | 2106.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-24 09:15:00 | 2060.00 | 2085.80 | 2095.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 2087.10 | 2084.51 | 2092.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 13:00:00 | 2087.10 | 2084.51 | 2092.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 2083.10 | 2084.23 | 2091.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 15:15:00 | 2070.00 | 2083.64 | 2090.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 2092.50 | 2083.23 | 2089.08 | SL hit (close>static) qty=1.00 sl=2092.00 alert=retest2 |

### Cycle 146 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 2101.40 | 2092.38 | 2092.09 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2026-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 14:15:00 | 2086.20 | 2091.15 | 2091.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 09:15:00 | 2075.80 | 2087.06 | 2089.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 12:15:00 | 2070.00 | 2067.92 | 2075.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-30 13:00:00 | 2070.00 | 2067.92 | 2075.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 2070.90 | 2068.34 | 2073.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 2086.10 | 2068.34 | 2073.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 2034.40 | 2061.55 | 2069.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:30:00 | 2027.80 | 2056.84 | 2066.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:00:00 | 2013.50 | 2048.17 | 2061.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 1926.41 | 2002.20 | 2034.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 1912.82 | 2002.20 | 2034.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 12:15:00 | 1956.50 | 1953.25 | 1981.72 | SL hit (close>ema200) qty=0.50 sl=1953.25 alert=retest2 |

### Cycle 148 — BUY (started 2026-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-09 12:15:00 | 1970.50 | 1964.09 | 1963.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 10:15:00 | 1979.30 | 1969.90 | 1966.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 14:15:00 | 2001.50 | 2006.30 | 1994.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-13 15:00:00 | 2001.50 | 2006.30 | 1994.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 1993.00 | 2003.64 | 1993.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 2011.60 | 2003.64 | 1993.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 13:15:00 | 1990.70 | 2001.11 | 1996.67 | SL hit (close<static) qty=1.00 sl=1993.00 alert=retest2 |

### Cycle 149 — SELL (started 2026-04-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-15 15:15:00 | 1978.90 | 1993.08 | 1993.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-16 09:15:00 | 1975.00 | 1989.46 | 1991.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-16 15:15:00 | 1977.90 | 1976.85 | 1982.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-17 09:15:00 | 1976.20 | 1976.85 | 1982.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 1977.00 | 1976.88 | 1982.42 | EMA400 retest candle locked (from downside) |

### Cycle 150 — BUY (started 2026-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 14:15:00 | 1998.90 | 1985.68 | 1984.93 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 1980.00 | 1989.33 | 1989.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 11:15:00 | 1974.40 | 1986.34 | 1988.30 | Break + close below crossover candle low |

### Cycle 152 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 2025.00 | 1988.29 | 1987.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 2029.80 | 2009.92 | 2003.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 12:15:00 | 2045.00 | 2047.08 | 2033.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 12:45:00 | 2041.50 | 2047.08 | 2033.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 2051.30 | 2047.63 | 2036.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 2064.30 | 2047.63 | 2036.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 13:45:00 | 2053.90 | 2049.25 | 2047.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 14:30:00 | 2054.30 | 2048.18 | 2047.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 2078.00 | 2048.51 | 2047.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 2161.90 | 2138.56 | 2114.37 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-03 15:00:00 | 1764.00 | 2024-06-04 09:15:00 | 1730.75 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-06-21 11:30:00 | 1773.00 | 2024-06-27 13:15:00 | 1684.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-21 11:30:00 | 1773.00 | 2024-06-28 09:15:00 | 1757.00 | STOP_HIT | 0.50 | 0.90% |
| BUY | retest2 | 2024-07-03 10:00:00 | 1804.80 | 2024-07-05 09:15:00 | 1753.40 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2024-07-03 11:30:00 | 1805.40 | 2024-07-05 09:15:00 | 1753.40 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2024-07-03 12:15:00 | 1804.00 | 2024-07-05 09:15:00 | 1753.40 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2024-07-03 15:00:00 | 1803.95 | 2024-07-05 09:15:00 | 1753.40 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2024-07-10 10:15:00 | 1701.45 | 2024-07-12 09:15:00 | 1744.75 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2024-07-10 11:15:00 | 1706.85 | 2024-07-12 09:15:00 | 1744.75 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2024-07-22 09:30:00 | 1807.95 | 2024-08-01 14:15:00 | 1912.15 | STOP_HIT | 1.00 | 5.76% |
| BUY | retest2 | 2024-07-23 12:30:00 | 1816.70 | 2024-08-01 14:15:00 | 1912.15 | STOP_HIT | 1.00 | 5.25% |
| SELL | retest2 | 2024-08-07 15:15:00 | 1910.05 | 2024-08-08 09:15:00 | 1942.70 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2024-08-14 12:15:00 | 1967.95 | 2024-08-19 15:15:00 | 1950.20 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-08-14 12:45:00 | 1988.75 | 2024-08-19 15:15:00 | 1950.20 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2024-08-16 09:15:00 | 2004.60 | 2024-08-19 15:15:00 | 1950.20 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2024-08-19 11:45:00 | 1967.35 | 2024-08-19 15:15:00 | 1950.20 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-08-23 09:15:00 | 1930.20 | 2024-08-27 10:15:00 | 1942.75 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-08-26 11:00:00 | 1899.45 | 2024-08-27 10:15:00 | 1942.75 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-09-10 13:00:00 | 1917.95 | 2024-09-13 12:15:00 | 1921.45 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2024-09-10 13:30:00 | 1920.90 | 2024-09-13 12:15:00 | 1921.45 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2024-09-10 14:00:00 | 1913.95 | 2024-09-13 12:15:00 | 1921.45 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2024-09-11 11:15:00 | 1916.60 | 2024-09-13 12:15:00 | 1921.45 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2024-09-11 12:45:00 | 1896.60 | 2024-09-13 12:15:00 | 1921.45 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-09-19 13:15:00 | 1854.00 | 2024-09-20 09:15:00 | 1903.95 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2024-09-30 12:00:00 | 1880.10 | 2024-10-03 10:15:00 | 1786.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 12:00:00 | 1880.10 | 2024-10-04 09:15:00 | 1692.09 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-10-16 14:30:00 | 1874.05 | 2024-10-22 13:15:00 | 1879.10 | STOP_HIT | 1.00 | 0.27% |
| BUY | retest2 | 2024-10-16 15:00:00 | 1904.20 | 2024-10-22 13:15:00 | 1879.10 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-10-18 09:30:00 | 1881.95 | 2024-10-22 13:15:00 | 1879.10 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2024-10-28 09:30:00 | 1871.80 | 2024-10-28 10:15:00 | 1910.40 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2024-10-28 13:00:00 | 1877.10 | 2024-10-31 09:15:00 | 1927.20 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2024-10-28 13:45:00 | 1878.25 | 2024-10-31 09:15:00 | 1927.20 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2024-10-29 09:30:00 | 1869.50 | 2024-10-31 09:15:00 | 1927.20 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2024-11-05 11:15:00 | 1872.90 | 2024-11-11 09:15:00 | 1779.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-06 14:00:00 | 1870.10 | 2024-11-11 09:15:00 | 1776.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-06 14:45:00 | 1867.05 | 2024-11-11 09:15:00 | 1773.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 09:15:00 | 1872.55 | 2024-11-11 09:15:00 | 1778.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 11:15:00 | 1867.90 | 2024-11-11 09:15:00 | 1774.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 12:00:00 | 1869.05 | 2024-11-11 09:15:00 | 1775.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-05 11:15:00 | 1872.90 | 2024-11-14 09:15:00 | 1685.61 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-06 14:00:00 | 1870.10 | 2024-11-14 09:15:00 | 1683.09 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-06 14:45:00 | 1867.05 | 2024-11-14 09:15:00 | 1680.35 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-07 09:15:00 | 1872.55 | 2024-11-14 09:15:00 | 1685.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-07 11:15:00 | 1867.90 | 2024-11-14 09:15:00 | 1681.11 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-07 12:00:00 | 1869.05 | 2024-11-14 09:15:00 | 1682.14 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-01-01 09:30:00 | 1851.25 | 2025-01-06 12:15:00 | 1834.60 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-01-01 10:00:00 | 1854.40 | 2025-01-06 12:15:00 | 1834.60 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-01-01 13:45:00 | 1850.75 | 2025-01-06 12:15:00 | 1834.60 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-01-01 14:45:00 | 1870.75 | 2025-01-06 12:15:00 | 1834.60 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-01-02 13:15:00 | 1852.75 | 2025-01-06 12:15:00 | 1834.60 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-01-02 13:45:00 | 1852.90 | 2025-01-06 12:15:00 | 1834.60 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-01-03 09:15:00 | 1870.25 | 2025-01-06 12:15:00 | 1834.60 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-01-06 11:15:00 | 1854.45 | 2025-01-06 12:15:00 | 1834.60 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest1 | 2025-01-13 09:15:00 | 1807.70 | 2025-01-16 09:15:00 | 1783.40 | STOP_HIT | 1.00 | 1.34% |
| SELL | retest1 | 2025-01-13 09:45:00 | 1815.25 | 2025-01-16 09:15:00 | 1783.40 | STOP_HIT | 1.00 | 1.75% |
| SELL | retest1 | 2025-01-13 10:30:00 | 1815.00 | 2025-01-16 09:15:00 | 1783.40 | STOP_HIT | 1.00 | 1.74% |
| SELL | retest1 | 2025-01-13 12:15:00 | 1812.30 | 2025-01-16 09:15:00 | 1783.40 | STOP_HIT | 1.00 | 1.59% |
| SELL | retest2 | 2025-01-17 12:15:00 | 1759.20 | 2025-01-20 13:15:00 | 1796.05 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-01-20 09:45:00 | 1757.75 | 2025-01-20 13:15:00 | 1796.05 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2025-01-22 14:00:00 | 1805.00 | 2025-01-24 14:15:00 | 1780.15 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-01-22 14:30:00 | 1802.40 | 2025-01-24 14:15:00 | 1780.15 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-01-23 10:15:00 | 1803.45 | 2025-01-24 14:15:00 | 1780.15 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-01-23 11:45:00 | 1805.75 | 2025-01-24 14:15:00 | 1780.15 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-01-30 11:15:00 | 1745.75 | 2025-01-30 12:15:00 | 1755.80 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-02-01 15:00:00 | 1789.90 | 2025-02-03 11:15:00 | 1729.55 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2025-02-07 09:15:00 | 1689.00 | 2025-02-11 13:15:00 | 1604.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 09:15:00 | 1689.00 | 2025-02-12 10:15:00 | 1520.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2025-02-18 09:30:00 | 1624.35 | 2025-02-19 12:15:00 | 1667.75 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest1 | 2025-02-18 10:15:00 | 1626.65 | 2025-02-19 12:15:00 | 1667.75 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2025-02-27 09:15:00 | 1544.85 | 2025-02-28 14:15:00 | 1653.80 | STOP_HIT | 1.00 | -7.05% |
| SELL | retest2 | 2025-02-27 15:15:00 | 1552.10 | 2025-02-28 14:15:00 | 1653.80 | STOP_HIT | 1.00 | -6.55% |
| SELL | retest2 | 2025-02-28 13:15:00 | 1573.45 | 2025-02-28 14:15:00 | 1653.80 | STOP_HIT | 1.00 | -5.11% |
| SELL | retest2 | 2025-02-28 14:00:00 | 1572.15 | 2025-02-28 14:15:00 | 1653.80 | STOP_HIT | 1.00 | -5.19% |
| SELL | retest2 | 2025-03-11 09:15:00 | 1546.85 | 2025-03-13 09:15:00 | 1469.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-11 09:15:00 | 1546.85 | 2025-03-13 10:15:00 | 1526.80 | STOP_HIT | 0.50 | 1.30% |
| SELL | retest2 | 2025-04-04 10:15:00 | 1514.90 | 2025-04-07 09:15:00 | 1439.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 11:30:00 | 1516.70 | 2025-04-07 09:15:00 | 1440.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 10:15:00 | 1514.90 | 2025-04-07 13:15:00 | 1499.50 | STOP_HIT | 0.50 | 1.02% |
| SELL | retest2 | 2025-04-04 11:30:00 | 1516.70 | 2025-04-07 13:15:00 | 1499.50 | STOP_HIT | 0.50 | 1.13% |
| SELL | retest2 | 2025-04-07 09:15:00 | 1445.50 | 2025-04-09 12:15:00 | 1531.80 | STOP_HIT | 1.00 | -5.97% |
| SELL | retest2 | 2025-04-07 15:15:00 | 1510.00 | 2025-04-09 12:15:00 | 1531.80 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-04-25 14:00:00 | 1567.50 | 2025-04-30 10:15:00 | 1602.40 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2025-04-25 14:45:00 | 1570.00 | 2025-04-30 10:15:00 | 1602.40 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-04-25 15:15:00 | 1570.00 | 2025-04-30 10:15:00 | 1602.40 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-05-08 13:30:00 | 1547.40 | 2025-05-12 11:15:00 | 1549.40 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-05-22 11:45:00 | 1689.00 | 2025-05-28 09:15:00 | 1669.50 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-05-22 12:30:00 | 1676.50 | 2025-05-28 09:15:00 | 1669.50 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-05-29 13:45:00 | 1677.00 | 2025-05-29 14:15:00 | 1693.40 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-07-10 13:15:00 | 1615.00 | 2025-07-14 12:15:00 | 1634.00 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-07-10 14:00:00 | 1614.70 | 2025-07-14 12:15:00 | 1634.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-07-10 14:45:00 | 1614.60 | 2025-07-14 12:15:00 | 1634.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-07-11 10:15:00 | 1614.40 | 2025-07-14 12:15:00 | 1634.00 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-07-16 14:45:00 | 1653.90 | 2025-08-01 10:15:00 | 1748.10 | STOP_HIT | 1.00 | 5.70% |
| BUY | retest2 | 2025-07-18 13:00:00 | 1660.00 | 2025-08-01 10:15:00 | 1748.10 | STOP_HIT | 1.00 | 5.31% |
| BUY | retest2 | 2025-07-21 09:45:00 | 1653.10 | 2025-08-01 10:15:00 | 1748.10 | STOP_HIT | 1.00 | 5.75% |
| BUY | retest2 | 2025-08-14 11:45:00 | 1724.70 | 2025-08-26 13:15:00 | 1730.10 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2025-08-14 12:45:00 | 1724.70 | 2025-08-26 13:15:00 | 1730.10 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2025-08-19 12:00:00 | 1724.50 | 2025-08-26 13:15:00 | 1730.10 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2025-08-19 13:15:00 | 1727.60 | 2025-08-26 13:15:00 | 1730.10 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2025-08-22 13:15:00 | 1739.80 | 2025-08-26 13:15:00 | 1730.10 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-08-26 09:45:00 | 1739.00 | 2025-08-26 13:15:00 | 1730.10 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-08-26 11:15:00 | 1739.50 | 2025-08-26 13:15:00 | 1730.10 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-10-10 11:30:00 | 1679.90 | 2025-10-13 13:15:00 | 1666.70 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-10-10 13:30:00 | 1678.40 | 2025-10-13 13:15:00 | 1666.70 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-10-23 09:15:00 | 1709.50 | 2025-10-28 11:15:00 | 1690.00 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-10-23 11:00:00 | 1712.90 | 2025-10-28 11:15:00 | 1690.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-10-23 14:00:00 | 1712.00 | 2025-10-28 11:15:00 | 1690.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-10-23 15:15:00 | 1710.20 | 2025-10-28 11:15:00 | 1690.00 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-10-30 09:15:00 | 1692.20 | 2025-10-30 11:15:00 | 1699.20 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-11-13 10:15:00 | 1837.00 | 2025-11-18 09:15:00 | 1799.50 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-11-14 09:30:00 | 1832.90 | 2025-11-18 09:15:00 | 1799.50 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-11-14 10:00:00 | 1833.70 | 2025-11-18 09:15:00 | 1799.50 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest1 | 2025-11-27 09:45:00 | 1786.80 | 2025-11-27 13:15:00 | 1770.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest1 | 2025-11-27 10:15:00 | 1787.10 | 2025-11-27 13:15:00 | 1770.00 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-12-03 10:45:00 | 1761.20 | 2025-12-03 13:15:00 | 1771.80 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-12-16 09:15:00 | 1787.70 | 2025-12-19 12:15:00 | 1789.90 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2025-12-16 10:00:00 | 1787.10 | 2025-12-19 12:15:00 | 1789.90 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-12-16 10:45:00 | 1790.60 | 2025-12-19 12:15:00 | 1789.90 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-12-16 11:15:00 | 1787.10 | 2025-12-19 12:15:00 | 1789.90 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-12-18 09:15:00 | 1783.30 | 2025-12-19 12:15:00 | 1789.90 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-12-19 09:45:00 | 1789.40 | 2025-12-19 12:15:00 | 1789.90 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-12-19 11:00:00 | 1788.60 | 2025-12-19 12:15:00 | 1789.90 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2025-12-24 10:15:00 | 1805.80 | 2025-12-29 12:15:00 | 1802.30 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-12-24 11:45:00 | 1807.10 | 2025-12-29 12:15:00 | 1802.30 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-12-26 10:15:00 | 1805.30 | 2025-12-29 12:15:00 | 1802.30 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2025-12-29 10:00:00 | 1808.40 | 2025-12-29 12:15:00 | 1802.30 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2026-01-02 10:15:00 | 1829.60 | 2026-01-12 09:15:00 | 1863.40 | STOP_HIT | 1.00 | 1.85% |
| SELL | retest2 | 2026-01-28 10:45:00 | 1852.70 | 2026-01-30 13:15:00 | 1879.80 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-01-28 11:15:00 | 1856.80 | 2026-01-30 13:15:00 | 1879.80 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-01-28 12:00:00 | 1854.90 | 2026-01-30 13:15:00 | 1879.80 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-01-28 13:15:00 | 1856.80 | 2026-01-30 13:15:00 | 1879.80 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2026-02-12 12:00:00 | 1906.00 | 2026-02-13 15:15:00 | 1889.80 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2026-02-13 09:30:00 | 1916.90 | 2026-02-13 15:15:00 | 1889.80 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest1 | 2026-03-12 13:00:00 | 2135.00 | 2026-03-13 09:15:00 | 2113.20 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2026-03-13 14:15:00 | 2120.00 | 2026-03-16 10:15:00 | 2095.10 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2026-03-16 09:30:00 | 2142.70 | 2026-03-16 10:15:00 | 2095.10 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2026-03-24 15:15:00 | 2070.00 | 2026-03-25 09:15:00 | 2092.50 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-04-01 11:30:00 | 2027.80 | 2026-04-02 09:15:00 | 1926.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 13:00:00 | 2013.50 | 2026-04-02 09:15:00 | 1912.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 11:30:00 | 2027.80 | 2026-04-06 12:15:00 | 1956.50 | STOP_HIT | 0.50 | 3.52% |
| SELL | retest2 | 2026-04-01 13:00:00 | 2013.50 | 2026-04-06 12:15:00 | 1956.50 | STOP_HIT | 0.50 | 2.83% |
| BUY | retest2 | 2026-04-15 09:15:00 | 2011.60 | 2026-04-15 13:15:00 | 1990.70 | STOP_HIT | 1.00 | -1.04% |
