# APL Apollo Tubes Ltd. (APLAPOLLO)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1950.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 78 |
| ALERT1 | 50 |
| ALERT2 | 50 |
| ALERT2_SKIP | 29 |
| ALERT3 | 151 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 64 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 65 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 67 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 45
- **Target hits / Stop hits / Partials:** 0 / 65 / 2
- **Avg / median % per leg:** 0.32% / -0.30%
- **Sum % (uncompounded):** 21.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 33 | 15 | 45.5% | 0 | 33 | 0 | 0.63% | 20.9% |
| BUY @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.11% | 0.1% |
| BUY @ 3rd Alert (retest2) | 32 | 14 | 43.8% | 0 | 32 | 0 | 0.65% | 20.8% |
| SELL (all) | 34 | 7 | 20.6% | 0 | 32 | 2 | 0.01% | 0.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 34 | 7 | 20.6% | 0 | 32 | 2 | 0.01% | 0.2% |
| retest1 (combined) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.11% | 0.1% |
| retest2 (combined) | 66 | 21 | 31.8% | 0 | 64 | 2 | 0.32% | 21.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 10:15:00 | 1830.00 | 1834.37 | 1834.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 11:15:00 | 1822.00 | 1831.90 | 1833.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 14:15:00 | 1829.80 | 1827.25 | 1830.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 14:15:00 | 1829.80 | 1827.25 | 1830.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 1829.80 | 1827.25 | 1830.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:45:00 | 1826.00 | 1827.25 | 1830.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1807.30 | 1822.90 | 1827.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 10:15:00 | 1801.60 | 1822.90 | 1827.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 12:30:00 | 1802.20 | 1812.65 | 1821.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 13:00:00 | 1801.30 | 1812.65 | 1821.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 14:00:00 | 1802.60 | 1810.64 | 1819.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 1811.60 | 1810.83 | 1819.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 15:00:00 | 1811.60 | 1810.83 | 1819.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 1807.90 | 1803.29 | 1809.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 1807.90 | 1803.29 | 1809.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 1825.10 | 1808.18 | 1810.97 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-03 11:15:00 | 1825.00 | 1813.90 | 1813.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 11:15:00 | 1825.00 | 1813.90 | 1813.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 09:15:00 | 1861.30 | 1825.37 | 1819.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 10:15:00 | 1902.10 | 1904.35 | 1883.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 10:45:00 | 1905.50 | 1904.35 | 1883.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 1893.30 | 1907.32 | 1895.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 11:00:00 | 1893.30 | 1907.32 | 1895.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 11:15:00 | 1890.70 | 1904.00 | 1895.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 12:00:00 | 1890.70 | 1904.00 | 1895.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 12:15:00 | 1885.70 | 1900.34 | 1894.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 12:45:00 | 1885.00 | 1900.34 | 1894.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 1893.00 | 1893.16 | 1892.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:00:00 | 1893.00 | 1893.16 | 1892.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 1895.50 | 1893.63 | 1892.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 15:00:00 | 1895.80 | 1894.32 | 1893.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 15:15:00 | 1887.40 | 1892.94 | 1892.71 | SL hit (close<static) qty=1.00 sl=1888.60 alert=retest2 |

### Cycle 3 — SELL (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 14:15:00 | 1888.40 | 1893.01 | 1893.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 15:15:00 | 1881.50 | 1890.71 | 1892.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 1846.20 | 1845.20 | 1858.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 1846.20 | 1845.20 | 1858.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1853.30 | 1842.75 | 1849.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 1853.30 | 1842.75 | 1849.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 1852.00 | 1844.60 | 1849.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 1859.30 | 1844.60 | 1849.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 1847.50 | 1845.18 | 1849.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:30:00 | 1843.10 | 1844.17 | 1848.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 1794.90 | 1789.70 | 1789.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 1794.90 | 1789.70 | 1789.03 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 11:15:00 | 1778.60 | 1790.66 | 1791.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 12:15:00 | 1762.80 | 1781.83 | 1786.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 12:15:00 | 1742.00 | 1737.04 | 1750.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 12:15:00 | 1742.00 | 1737.04 | 1750.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 1742.00 | 1737.04 | 1750.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:30:00 | 1750.00 | 1737.04 | 1750.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 1762.30 | 1743.91 | 1751.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 15:00:00 | 1762.30 | 1743.91 | 1751.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 1761.70 | 1747.47 | 1752.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 1746.60 | 1747.47 | 1752.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 1747.00 | 1742.19 | 1746.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:15:00 | 1758.70 | 1742.19 | 1746.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1737.20 | 1741.19 | 1745.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 11:15:00 | 1731.90 | 1740.81 | 1744.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:00:00 | 1734.00 | 1739.45 | 1743.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:30:00 | 1734.40 | 1738.20 | 1742.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 13:00:00 | 1733.20 | 1738.20 | 1742.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 1724.60 | 1732.14 | 1738.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 11:30:00 | 1720.60 | 1727.89 | 1735.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 11:15:00 | 1738.80 | 1730.98 | 1730.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 11:15:00 | 1738.80 | 1730.98 | 1730.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 14:15:00 | 1741.50 | 1735.45 | 1733.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 10:15:00 | 1726.30 | 1734.90 | 1733.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 10:15:00 | 1726.30 | 1734.90 | 1733.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 1726.30 | 1734.90 | 1733.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:00:00 | 1726.30 | 1734.90 | 1733.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 11:15:00 | 1722.20 | 1732.36 | 1732.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 12:15:00 | 1705.00 | 1726.89 | 1730.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 11:15:00 | 1719.50 | 1716.46 | 1722.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-10 12:00:00 | 1719.50 | 1716.46 | 1722.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 1722.30 | 1717.63 | 1722.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:30:00 | 1723.10 | 1717.63 | 1722.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 1724.70 | 1719.04 | 1722.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:45:00 | 1723.60 | 1719.04 | 1722.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 1718.30 | 1718.89 | 1722.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 15:15:00 | 1715.00 | 1718.89 | 1722.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 09:45:00 | 1715.00 | 1716.67 | 1720.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 12:15:00 | 1719.20 | 1707.49 | 1706.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 1719.20 | 1707.49 | 1706.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 13:15:00 | 1721.70 | 1710.33 | 1708.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 1711.00 | 1713.72 | 1710.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 1711.00 | 1713.72 | 1710.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 1711.00 | 1713.72 | 1710.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:30:00 | 1705.50 | 1713.72 | 1710.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 1710.30 | 1713.03 | 1710.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:00:00 | 1710.30 | 1713.03 | 1710.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 1712.90 | 1713.01 | 1710.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 14:45:00 | 1717.20 | 1713.74 | 1711.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 15:15:00 | 1718.70 | 1713.74 | 1711.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 10:15:00 | 1717.90 | 1714.98 | 1712.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 10:15:00 | 1705.90 | 1713.17 | 1712.02 | SL hit (close<static) qty=1.00 sl=1706.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 11:15:00 | 1702.00 | 1710.93 | 1711.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 14:15:00 | 1693.40 | 1705.21 | 1708.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 10:15:00 | 1684.90 | 1674.97 | 1685.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 10:15:00 | 1684.90 | 1674.97 | 1685.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 1684.90 | 1674.97 | 1685.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:00:00 | 1684.90 | 1674.97 | 1685.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1695.40 | 1679.06 | 1686.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:45:00 | 1696.80 | 1679.06 | 1686.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 1696.20 | 1682.49 | 1687.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 14:15:00 | 1687.80 | 1684.03 | 1687.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 14:45:00 | 1688.70 | 1686.04 | 1688.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:15:00 | 1669.70 | 1687.81 | 1688.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 1693.70 | 1676.01 | 1675.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 09:15:00 | 1693.70 | 1676.01 | 1675.31 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 1608.40 | 1668.81 | 1673.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 12:15:00 | 1573.10 | 1628.56 | 1652.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 1522.20 | 1516.27 | 1549.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:00:00 | 1522.20 | 1516.27 | 1549.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1552.00 | 1524.31 | 1542.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:00:00 | 1552.00 | 1524.31 | 1542.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 1561.40 | 1531.73 | 1544.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:30:00 | 1560.00 | 1531.73 | 1544.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 14:15:00 | 1566.50 | 1550.84 | 1550.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 15:15:00 | 1568.90 | 1554.45 | 1552.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 14:15:00 | 1586.80 | 1598.51 | 1586.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 14:15:00 | 1586.80 | 1598.51 | 1586.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 1586.80 | 1598.51 | 1586.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 1586.80 | 1598.51 | 1586.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 1594.10 | 1597.63 | 1587.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:15:00 | 1593.00 | 1597.63 | 1587.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 1604.20 | 1598.94 | 1588.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:30:00 | 1584.10 | 1598.94 | 1588.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 1588.60 | 1596.65 | 1590.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 1588.60 | 1596.65 | 1590.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 1583.50 | 1594.02 | 1589.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:30:00 | 1585.80 | 1594.02 | 1589.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1587.00 | 1592.62 | 1589.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 09:15:00 | 1608.70 | 1591.27 | 1588.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 11:15:00 | 1579.50 | 1597.64 | 1597.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 1579.50 | 1597.64 | 1597.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 14:15:00 | 1568.40 | 1586.55 | 1592.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 13:15:00 | 1570.50 | 1569.66 | 1579.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 14:00:00 | 1570.50 | 1569.66 | 1579.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 1596.70 | 1575.07 | 1580.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 1596.70 | 1575.07 | 1580.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 1591.30 | 1578.32 | 1581.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 1602.50 | 1578.32 | 1581.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 10:15:00 | 1592.30 | 1584.71 | 1584.40 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 1581.00 | 1584.76 | 1584.80 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 09:15:00 | 1597.10 | 1587.23 | 1585.92 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 09:15:00 | 1576.30 | 1585.33 | 1585.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 10:15:00 | 1567.80 | 1581.82 | 1584.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 12:15:00 | 1580.80 | 1580.54 | 1583.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-12 13:00:00 | 1580.80 | 1580.54 | 1583.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 1584.90 | 1581.41 | 1583.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:00:00 | 1584.90 | 1581.41 | 1583.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 1580.50 | 1581.23 | 1582.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:30:00 | 1585.10 | 1581.23 | 1582.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 1600.10 | 1585.13 | 1584.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 10:15:00 | 1605.00 | 1589.10 | 1586.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 12:15:00 | 1589.20 | 1589.30 | 1586.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 12:30:00 | 1591.50 | 1589.30 | 1586.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 1585.10 | 1588.46 | 1586.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:45:00 | 1584.20 | 1588.46 | 1586.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 1582.60 | 1587.29 | 1586.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 1582.60 | 1587.29 | 1586.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 1583.00 | 1586.43 | 1586.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 1593.80 | 1586.43 | 1586.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 1594.20 | 1587.98 | 1586.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 14:30:00 | 1601.50 | 1594.93 | 1590.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 10:15:00 | 1631.20 | 1642.16 | 1642.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 1631.20 | 1642.16 | 1642.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 15:15:00 | 1623.90 | 1634.98 | 1638.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1619.50 | 1610.41 | 1620.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 1619.50 | 1610.41 | 1620.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 1619.50 | 1610.41 | 1620.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 1619.50 | 1610.41 | 1620.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 1627.00 | 1613.73 | 1621.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 1631.00 | 1613.73 | 1621.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 1617.40 | 1614.46 | 1620.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 12:30:00 | 1623.90 | 1614.46 | 1620.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 1608.10 | 1613.19 | 1619.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 13:45:00 | 1616.30 | 1613.19 | 1619.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1624.00 | 1612.26 | 1617.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 1624.00 | 1612.26 | 1617.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1635.00 | 1616.81 | 1619.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 1635.00 | 1616.81 | 1619.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 1629.10 | 1619.27 | 1619.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:45:00 | 1635.00 | 1619.27 | 1619.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — BUY (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 12:15:00 | 1631.00 | 1621.61 | 1620.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 13:15:00 | 1634.20 | 1624.13 | 1622.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 1653.90 | 1654.41 | 1644.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 11:00:00 | 1653.90 | 1654.41 | 1644.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 1662.40 | 1668.85 | 1663.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:00:00 | 1662.40 | 1668.85 | 1663.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 1653.30 | 1665.74 | 1662.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:00:00 | 1653.30 | 1665.74 | 1662.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 1653.60 | 1663.31 | 1661.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:30:00 | 1652.50 | 1663.31 | 1661.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 1664.40 | 1664.25 | 1662.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:00:00 | 1664.40 | 1664.25 | 1662.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1677.10 | 1667.42 | 1664.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 13:45:00 | 1684.80 | 1677.10 | 1670.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 13:00:00 | 1685.80 | 1682.11 | 1676.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 14:00:00 | 1685.00 | 1689.85 | 1684.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 15:15:00 | 1682.90 | 1683.86 | 1683.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2025-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 15:15:00 | 1682.90 | 1683.86 | 1683.97 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 12:15:00 | 1707.10 | 1687.56 | 1685.52 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 10:15:00 | 1691.20 | 1693.97 | 1693.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 13:15:00 | 1679.60 | 1690.30 | 1692.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 15:15:00 | 1690.00 | 1689.99 | 1691.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 15:15:00 | 1690.00 | 1689.99 | 1691.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 1690.00 | 1689.99 | 1691.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 1693.70 | 1689.99 | 1691.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1686.50 | 1689.29 | 1691.26 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2025-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 14:15:00 | 1695.80 | 1692.36 | 1692.20 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 09:15:00 | 1687.40 | 1691.69 | 1691.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 10:15:00 | 1673.70 | 1688.10 | 1690.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 13:15:00 | 1688.00 | 1684.60 | 1687.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 13:15:00 | 1688.00 | 1684.60 | 1687.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 1688.00 | 1684.60 | 1687.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:00:00 | 1688.00 | 1684.60 | 1687.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 1691.00 | 1685.88 | 1688.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 1691.00 | 1685.88 | 1688.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 1694.00 | 1687.50 | 1688.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:30:00 | 1694.50 | 1687.46 | 1688.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 1691.50 | 1688.27 | 1688.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 11:00:00 | 1691.50 | 1688.27 | 1688.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 1686.00 | 1687.82 | 1688.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 14:15:00 | 1680.60 | 1686.10 | 1687.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 09:30:00 | 1674.40 | 1680.60 | 1684.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 1678.30 | 1679.85 | 1681.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 11:15:00 | 1676.10 | 1681.21 | 1682.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 1685.90 | 1682.15 | 1682.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 1685.90 | 1682.15 | 1682.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 1685.20 | 1682.76 | 1682.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:30:00 | 1683.10 | 1682.76 | 1682.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 1676.50 | 1681.51 | 1682.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 15:15:00 | 1673.10 | 1680.63 | 1681.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 1689.40 | 1681.18 | 1681.76 | SL hit (close>static) qty=1.00 sl=1689.00 alert=retest2 |

### Cycle 26 — BUY (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 10:15:00 | 1688.50 | 1682.64 | 1682.38 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2025-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 11:15:00 | 1679.60 | 1682.03 | 1682.12 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2025-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 12:15:00 | 1683.70 | 1682.37 | 1682.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 13:15:00 | 1691.00 | 1684.09 | 1683.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 10:15:00 | 1683.10 | 1686.95 | 1685.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 10:15:00 | 1683.10 | 1686.95 | 1685.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 1683.10 | 1686.95 | 1685.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:00:00 | 1683.10 | 1686.95 | 1685.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 1672.60 | 1684.08 | 1683.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:45:00 | 1669.00 | 1684.08 | 1683.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — SELL (started 2025-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 12:15:00 | 1670.00 | 1681.26 | 1682.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 13:15:00 | 1667.90 | 1678.59 | 1681.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 1685.00 | 1674.42 | 1678.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 1685.00 | 1674.42 | 1678.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1685.00 | 1674.42 | 1678.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:30:00 | 1685.50 | 1674.42 | 1678.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 1685.00 | 1676.54 | 1678.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:30:00 | 1685.80 | 1676.54 | 1678.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2025-09-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 12:15:00 | 1697.70 | 1683.24 | 1681.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 15:15:00 | 1702.00 | 1690.18 | 1685.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-30 09:15:00 | 1688.80 | 1689.90 | 1685.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-30 10:00:00 | 1688.80 | 1689.90 | 1685.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 1683.70 | 1688.16 | 1685.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:00:00 | 1683.70 | 1688.16 | 1685.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 1690.50 | 1688.63 | 1686.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:30:00 | 1691.20 | 1688.63 | 1686.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 1687.90 | 1688.49 | 1686.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 14:00:00 | 1687.90 | 1688.49 | 1686.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 1683.30 | 1687.45 | 1686.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 09:30:00 | 1696.00 | 1687.76 | 1686.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 10:15:00 | 1696.10 | 1687.76 | 1686.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 11:00:00 | 1696.50 | 1689.51 | 1687.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 1724.20 | 1732.43 | 1732.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 1724.20 | 1732.43 | 1732.60 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 09:15:00 | 1742.80 | 1732.77 | 1732.62 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 1721.40 | 1734.69 | 1736.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 1707.00 | 1729.15 | 1733.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1730.30 | 1725.99 | 1729.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 1730.30 | 1725.99 | 1729.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1730.30 | 1725.99 | 1729.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 1728.90 | 1725.99 | 1729.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1742.40 | 1729.27 | 1730.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 1742.40 | 1729.27 | 1730.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 1736.80 | 1730.78 | 1731.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 12:15:00 | 1735.40 | 1730.78 | 1731.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 12:45:00 | 1733.70 | 1731.04 | 1731.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 13:15:00 | 1735.60 | 1731.95 | 1731.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 13:15:00 | 1735.60 | 1731.95 | 1731.84 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 09:15:00 | 1722.20 | 1730.27 | 1731.13 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 10:15:00 | 1735.30 | 1730.32 | 1730.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 12:15:00 | 1753.00 | 1735.91 | 1732.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 13:15:00 | 1747.60 | 1750.14 | 1744.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 14:00:00 | 1747.60 | 1750.14 | 1744.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 1745.50 | 1749.21 | 1744.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 15:15:00 | 1754.00 | 1749.21 | 1744.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 10:45:00 | 1749.80 | 1753.71 | 1751.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 11:45:00 | 1753.20 | 1752.97 | 1751.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 14:00:00 | 1750.50 | 1750.91 | 1750.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 1754.00 | 1751.53 | 1750.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:45:00 | 1750.00 | 1751.53 | 1750.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 1769.80 | 1756.01 | 1752.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 11:15:00 | 1776.80 | 1759.09 | 1754.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:45:00 | 1775.50 | 1765.26 | 1760.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 09:30:00 | 1780.00 | 1792.40 | 1783.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 10:30:00 | 1781.90 | 1791.92 | 1784.07 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 1781.20 | 1790.44 | 1784.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:00:00 | 1781.20 | 1790.44 | 1784.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 1784.00 | 1789.16 | 1784.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:30:00 | 1778.60 | 1789.16 | 1784.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 1783.50 | 1788.02 | 1784.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 1797.00 | 1787.22 | 1784.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 10:15:00 | 1778.90 | 1792.57 | 1793.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 1778.90 | 1792.57 | 1793.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 11:15:00 | 1765.60 | 1787.18 | 1791.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 09:15:00 | 1793.90 | 1784.40 | 1787.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 1793.90 | 1784.40 | 1787.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1793.90 | 1784.40 | 1787.56 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 11:15:00 | 1807.00 | 1792.04 | 1790.68 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2025-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 15:15:00 | 1783.20 | 1790.38 | 1790.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 1779.60 | 1788.22 | 1789.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 1790.70 | 1788.72 | 1789.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 10:15:00 | 1790.70 | 1788.72 | 1789.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 1790.70 | 1788.72 | 1789.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:00:00 | 1790.70 | 1788.72 | 1789.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 1793.60 | 1789.70 | 1790.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:30:00 | 1797.80 | 1789.70 | 1790.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 13:15:00 | 1801.40 | 1792.13 | 1791.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 14:15:00 | 1804.00 | 1794.51 | 1792.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 09:15:00 | 1791.60 | 1795.25 | 1793.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 09:15:00 | 1791.60 | 1795.25 | 1793.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1791.60 | 1795.25 | 1793.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 1791.80 | 1795.25 | 1793.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1796.20 | 1795.44 | 1793.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 11:30:00 | 1801.50 | 1797.13 | 1794.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 14:15:00 | 1790.00 | 1794.68 | 1795.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 14:15:00 | 1790.00 | 1794.68 | 1795.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 10:15:00 | 1781.70 | 1791.29 | 1793.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 12:15:00 | 1796.30 | 1791.45 | 1792.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 12:15:00 | 1796.30 | 1791.45 | 1792.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 1796.30 | 1791.45 | 1792.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 12:45:00 | 1795.20 | 1791.45 | 1792.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 1792.70 | 1791.70 | 1792.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 13:45:00 | 1796.10 | 1791.70 | 1792.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 1795.60 | 1792.48 | 1793.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 15:00:00 | 1795.60 | 1792.48 | 1793.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 1796.90 | 1793.36 | 1793.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:15:00 | 1791.50 | 1793.36 | 1793.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 09:15:00 | 1795.70 | 1793.83 | 1793.72 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 11:15:00 | 1790.10 | 1793.18 | 1793.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 12:15:00 | 1785.60 | 1791.66 | 1792.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 11:15:00 | 1773.80 | 1771.45 | 1780.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-14 12:15:00 | 1776.60 | 1771.45 | 1780.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1783.70 | 1772.35 | 1777.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:30:00 | 1786.00 | 1772.35 | 1777.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 1775.00 | 1772.88 | 1776.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:30:00 | 1771.30 | 1773.22 | 1775.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 12:30:00 | 1768.50 | 1772.11 | 1774.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 13:15:00 | 1728.90 | 1718.77 | 1718.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 1728.90 | 1718.77 | 1718.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 1734.10 | 1721.84 | 1720.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 1717.40 | 1722.42 | 1720.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 1717.40 | 1722.42 | 1720.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 1717.40 | 1722.42 | 1720.71 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2025-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 12:15:00 | 1716.00 | 1721.44 | 1721.87 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 1731.20 | 1722.16 | 1721.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 11:15:00 | 1747.40 | 1730.90 | 1728.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 15:15:00 | 1769.00 | 1770.37 | 1761.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 09:15:00 | 1766.50 | 1770.37 | 1761.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 1748.90 | 1766.08 | 1760.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 1748.90 | 1766.08 | 1760.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 1747.50 | 1762.36 | 1759.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:00:00 | 1747.50 | 1762.36 | 1759.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 1738.30 | 1757.55 | 1757.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:00:00 | 1738.30 | 1757.55 | 1757.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 1732.10 | 1752.46 | 1754.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 09:15:00 | 1722.10 | 1734.27 | 1739.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 1727.30 | 1725.44 | 1730.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 1727.30 | 1725.44 | 1730.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1727.30 | 1725.44 | 1730.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:45:00 | 1728.50 | 1725.44 | 1730.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 1722.60 | 1724.87 | 1730.13 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 15:15:00 | 1736.00 | 1732.56 | 1732.41 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 1730.10 | 1732.07 | 1732.20 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2025-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 14:15:00 | 1741.00 | 1732.55 | 1732.09 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 09:15:00 | 1721.60 | 1731.00 | 1731.84 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 10:15:00 | 1740.90 | 1732.98 | 1732.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-17 12:15:00 | 1770.90 | 1741.37 | 1736.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 11:15:00 | 1861.30 | 1865.25 | 1850.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 11:45:00 | 1858.90 | 1865.25 | 1850.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 1855.20 | 1862.41 | 1851.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:30:00 | 1852.60 | 1862.41 | 1851.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 1861.40 | 1862.21 | 1852.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:30:00 | 1853.80 | 1862.21 | 1852.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1863.30 | 1860.95 | 1853.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:15:00 | 1871.10 | 1860.95 | 1853.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 11:00:00 | 1876.60 | 1864.08 | 1855.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 09:15:00 | 1931.40 | 1944.32 | 1945.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 1931.40 | 1944.32 | 1945.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 1925.00 | 1940.46 | 1943.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 1906.50 | 1885.52 | 1899.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 14:15:00 | 1906.50 | 1885.52 | 1899.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 1906.50 | 1885.52 | 1899.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 15:00:00 | 1906.50 | 1885.52 | 1899.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1905.00 | 1889.41 | 1899.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 1905.00 | 1889.41 | 1899.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 1900.40 | 1891.87 | 1899.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:00:00 | 1900.40 | 1891.87 | 1899.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 1884.40 | 1890.37 | 1897.72 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 09:15:00 | 1921.50 | 1902.85 | 1901.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 11:15:00 | 1943.80 | 1916.14 | 1908.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 1936.50 | 1938.73 | 1927.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 13:30:00 | 1932.00 | 1938.73 | 1927.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1934.10 | 1938.61 | 1930.62 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2026-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 14:15:00 | 1921.50 | 1927.01 | 1927.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 1911.60 | 1920.83 | 1923.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1946.20 | 1898.13 | 1904.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 1946.20 | 1898.13 | 1904.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1946.20 | 1898.13 | 1904.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 1953.10 | 1898.13 | 1904.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1935.60 | 1905.63 | 1907.69 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 1947.90 | 1914.08 | 1911.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 13:15:00 | 1954.70 | 1926.18 | 1917.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 15:15:00 | 1997.80 | 1998.45 | 1970.97 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-27 09:15:00 | 2052.70 | 1998.45 | 1970.97 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 2054.90 | 2073.61 | 2059.30 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-29 13:15:00 | 2054.90 | 2073.61 | 2059.30 | SL hit (close<ema400) qty=1.00 sl=2059.30 alert=retest1 |

### Cycle 57 — SELL (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 13:15:00 | 2049.90 | 2054.72 | 2054.92 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 12:15:00 | 2066.20 | 2055.05 | 2054.36 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 2046.10 | 2053.25 | 2053.66 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 2080.20 | 2058.70 | 2055.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 2112.90 | 2072.66 | 2062.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 2159.50 | 2160.72 | 2133.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 2159.50 | 2160.72 | 2133.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 2272.80 | 2278.70 | 2261.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:45:00 | 2269.40 | 2278.70 | 2261.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 2239.40 | 2268.65 | 2260.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:00:00 | 2239.40 | 2268.65 | 2260.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 2249.20 | 2264.76 | 2259.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 11:45:00 | 2257.90 | 2263.75 | 2259.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 2257.80 | 2258.40 | 2258.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 2248.70 | 2256.46 | 2257.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 09:15:00 | 2248.70 | 2256.46 | 2257.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 10:15:00 | 2231.20 | 2251.41 | 2254.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 14:15:00 | 2224.30 | 2219.74 | 2230.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 15:00:00 | 2224.30 | 2219.74 | 2230.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 2249.70 | 2226.06 | 2231.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:30:00 | 2261.10 | 2226.06 | 2231.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 2249.30 | 2230.71 | 2233.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:00:00 | 2249.30 | 2230.71 | 2233.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 12:15:00 | 2250.00 | 2237.57 | 2236.28 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 2208.60 | 2231.40 | 2234.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 2203.50 | 2218.77 | 2226.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 09:15:00 | 2218.50 | 2214.74 | 2223.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 09:15:00 | 2218.50 | 2214.74 | 2223.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 2218.50 | 2214.74 | 2223.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:15:00 | 2224.90 | 2214.74 | 2223.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 2218.60 | 2215.51 | 2222.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:30:00 | 2227.00 | 2215.51 | 2222.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 2211.00 | 2214.61 | 2221.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 12:30:00 | 2203.10 | 2212.39 | 2220.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 14:15:00 | 2202.80 | 2212.31 | 2219.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 09:30:00 | 2206.20 | 2201.00 | 2211.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:30:00 | 2204.40 | 2194.73 | 2202.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 2204.70 | 2196.72 | 2202.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 10:30:00 | 2203.90 | 2196.72 | 2202.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 2201.00 | 2197.58 | 2202.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 12:15:00 | 2199.10 | 2197.58 | 2202.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 2222.30 | 2201.48 | 2201.67 | SL hit (close>static) qty=1.00 sl=2205.30 alert=retest2 |

### Cycle 64 — BUY (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 11:15:00 | 2224.40 | 2206.06 | 2203.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 2231.70 | 2212.62 | 2207.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 10:15:00 | 2214.00 | 2216.04 | 2210.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 10:15:00 | 2214.00 | 2216.04 | 2210.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 2214.00 | 2216.04 | 2210.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:45:00 | 2209.50 | 2216.04 | 2210.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 2214.20 | 2215.68 | 2210.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:30:00 | 2212.00 | 2215.68 | 2210.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 2217.00 | 2215.94 | 2211.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:30:00 | 2212.80 | 2215.94 | 2211.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 2236.60 | 2233.37 | 2225.87 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 2164.50 | 2213.99 | 2219.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 11:15:00 | 2127.80 | 2186.23 | 2205.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 2160.50 | 2149.18 | 2176.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 10:00:00 | 2160.50 | 2149.18 | 2176.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 2172.80 | 2155.43 | 2166.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:45:00 | 2172.90 | 2155.43 | 2166.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 2181.50 | 2160.64 | 2167.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:45:00 | 2161.00 | 2164.75 | 2168.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 10:15:00 | 2052.95 | 2130.20 | 2150.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 2152.10 | 2112.15 | 2128.65 | SL hit (close>ema200) qty=0.50 sl=2112.15 alert=retest2 |

### Cycle 66 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 1994.70 | 1949.53 | 1946.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 11:15:00 | 2005.20 | 1968.64 | 1956.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1962.90 | 1987.82 | 1972.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1962.90 | 1987.82 | 1972.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1962.90 | 1987.82 | 1972.86 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 1935.60 | 1965.02 | 1965.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1926.90 | 1957.40 | 1962.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1981.80 | 1958.77 | 1961.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 1981.80 | 1958.77 | 1961.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1981.80 | 1958.77 | 1961.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:45:00 | 1983.00 | 1958.77 | 1961.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 1985.00 | 1964.02 | 1963.92 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 1932.40 | 1965.44 | 1966.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 11:15:00 | 1913.40 | 1947.65 | 1957.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1962.10 | 1924.00 | 1935.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 12:15:00 | 1962.10 | 1924.00 | 1935.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1962.10 | 1924.00 | 1935.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 1962.10 | 1924.00 | 1935.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1982.30 | 1935.66 | 1939.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:45:00 | 1977.00 | 1935.66 | 1939.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 1971.90 | 1942.91 | 1942.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 2012.80 | 1963.62 | 1952.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1976.80 | 1989.86 | 1975.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1976.80 | 1989.86 | 1975.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1976.80 | 1989.86 | 1975.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 12:15:00 | 2004.70 | 1992.02 | 1978.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 1957.20 | 1987.37 | 1982.12 | SL hit (close<static) qty=1.00 sl=1965.70 alert=retest2 |

### Cycle 71 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 1936.90 | 1977.28 | 1978.01 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 1991.10 | 1974.70 | 1973.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 2000.30 | 1979.82 | 1976.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-01 13:15:00 | 1955.00 | 1974.86 | 1974.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 13:15:00 | 1955.00 | 1974.86 | 1974.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 13:15:00 | 1955.00 | 1974.86 | 1974.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 13:30:00 | 1927.10 | 1974.86 | 1974.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 14:15:00 | 1930.70 | 1966.03 | 1970.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 1846.30 | 1937.44 | 1956.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 15:15:00 | 1907.00 | 1903.70 | 1926.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-06 09:15:00 | 1878.20 | 1903.70 | 1926.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 13:15:00 | 1913.00 | 1902.44 | 1916.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 14:15:00 | 1922.80 | 1902.44 | 1916.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 14:15:00 | 1917.30 | 1905.41 | 1916.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:15:00 | 1885.00 | 1907.73 | 1916.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 1983.80 | 1910.51 | 1909.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 1983.80 | 1910.51 | 1909.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 1997.60 | 1927.93 | 1917.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 10:15:00 | 2028.00 | 2029.04 | 2003.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 11:00:00 | 2028.00 | 2029.04 | 2003.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1999.40 | 2040.61 | 2022.91 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 14:15:00 | 1980.70 | 2012.32 | 2014.48 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2026-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 12:15:00 | 2033.00 | 2017.05 | 2015.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 14:15:00 | 2041.90 | 2023.63 | 2018.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 11:15:00 | 2025.90 | 2030.60 | 2024.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 11:15:00 | 2025.90 | 2030.60 | 2024.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 2025.90 | 2030.60 | 2024.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 12:00:00 | 2025.90 | 2030.60 | 2024.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 2036.70 | 2031.82 | 2025.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 12:30:00 | 2022.30 | 2031.82 | 2025.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 2102.40 | 2120.84 | 2104.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 13:00:00 | 2102.40 | 2120.84 | 2104.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 2104.30 | 2117.54 | 2104.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 13:30:00 | 2102.70 | 2117.54 | 2104.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 2105.50 | 2115.13 | 2104.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 14:30:00 | 2104.10 | 2115.13 | 2104.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 2114.00 | 2114.90 | 2105.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:15:00 | 2113.60 | 2114.90 | 2105.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 2118.80 | 2115.68 | 2106.35 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 2048.20 | 2096.75 | 2103.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 13:15:00 | 2044.90 | 2074.12 | 2090.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 2016.90 | 2016.16 | 2040.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 2016.90 | 2016.16 | 2040.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 2016.90 | 2016.16 | 2040.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:15:00 | 2002.90 | 2015.43 | 2037.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 11:15:00 | 1902.76 | 1945.10 | 1968.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-04 15:15:00 | 1877.00 | 1876.45 | 1907.56 | SL hit (close>ema200) qty=0.50 sl=1876.45 alert=retest2 |

### Cycle 78 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 1918.00 | 1894.89 | 1892.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 1957.10 | 1907.33 | 1898.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 12:15:00 | 1948.70 | 1953.88 | 1936.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 13:00:00 | 1948.70 | 1953.88 | 1936.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 09:45:00 | 1723.40 | 2025-05-29 10:15:00 | 1830.00 | STOP_HIT | 1.00 | 6.19% |
| BUY | retest2 | 2025-05-12 11:00:00 | 1711.30 | 2025-05-29 10:15:00 | 1830.00 | STOP_HIT | 1.00 | 6.94% |
| SELL | retest2 | 2025-05-30 10:15:00 | 1801.60 | 2025-06-03 11:15:00 | 1825.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-05-30 12:30:00 | 1802.20 | 2025-06-03 11:15:00 | 1825.00 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-05-30 13:00:00 | 1801.30 | 2025-06-03 11:15:00 | 1825.00 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-05-30 14:00:00 | 1802.60 | 2025-06-03 11:15:00 | 1825.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-06-10 15:00:00 | 1895.80 | 2025-06-10 15:15:00 | 1887.40 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-06-11 09:15:00 | 1906.50 | 2025-06-11 13:15:00 | 1887.50 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-06-11 12:00:00 | 1896.80 | 2025-06-11 13:15:00 | 1887.50 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-06-11 12:30:00 | 1896.00 | 2025-06-11 13:15:00 | 1887.50 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-06-17 10:30:00 | 1843.10 | 2025-06-24 09:15:00 | 1794.90 | STOP_HIT | 1.00 | 2.62% |
| SELL | retest2 | 2025-07-03 11:15:00 | 1731.90 | 2025-07-08 11:15:00 | 1738.80 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-07-03 12:00:00 | 1734.00 | 2025-07-08 11:15:00 | 1738.80 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-07-03 12:30:00 | 1734.40 | 2025-07-08 11:15:00 | 1738.80 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-07-03 13:00:00 | 1733.20 | 2025-07-08 11:15:00 | 1738.80 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-07-04 11:30:00 | 1720.60 | 2025-07-08 11:15:00 | 1738.80 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-07-10 15:15:00 | 1715.00 | 2025-07-15 12:15:00 | 1719.20 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-07-11 09:45:00 | 1715.00 | 2025-07-15 12:15:00 | 1719.20 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-07-16 14:45:00 | 1717.20 | 2025-07-17 10:15:00 | 1705.90 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-07-16 15:15:00 | 1718.70 | 2025-07-17 10:15:00 | 1705.90 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-07-17 10:15:00 | 1717.90 | 2025-07-17 10:15:00 | 1705.90 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-07-21 14:15:00 | 1687.80 | 2025-07-24 09:15:00 | 1693.70 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-07-21 14:45:00 | 1688.70 | 2025-07-24 09:15:00 | 1693.70 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-07-22 09:15:00 | 1669.70 | 2025-07-24 09:15:00 | 1693.70 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-08-05 09:15:00 | 1608.70 | 2025-08-06 11:15:00 | 1579.50 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-08-14 14:30:00 | 1601.50 | 2025-08-26 10:15:00 | 1631.20 | STOP_HIT | 1.00 | 1.85% |
| BUY | retest2 | 2025-09-08 13:45:00 | 1684.80 | 2025-09-11 15:15:00 | 1682.90 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-09-09 13:00:00 | 1685.80 | 2025-09-11 15:15:00 | 1682.90 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2025-09-10 14:00:00 | 1685.00 | 2025-09-11 15:15:00 | 1682.90 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2025-09-22 14:15:00 | 1680.60 | 2025-09-25 09:15:00 | 1689.40 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-09-23 09:30:00 | 1674.40 | 2025-09-25 10:15:00 | 1688.50 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-09-24 09:15:00 | 1678.30 | 2025-09-25 10:15:00 | 1688.50 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-09-24 11:15:00 | 1676.10 | 2025-09-25 10:15:00 | 1688.50 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-09-24 15:15:00 | 1673.10 | 2025-09-25 10:15:00 | 1688.50 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-10-01 09:30:00 | 1696.00 | 2025-10-08 14:15:00 | 1724.20 | STOP_HIT | 1.00 | 1.66% |
| BUY | retest2 | 2025-10-01 10:15:00 | 1696.10 | 2025-10-08 14:15:00 | 1724.20 | STOP_HIT | 1.00 | 1.66% |
| BUY | retest2 | 2025-10-01 11:00:00 | 1696.50 | 2025-10-08 14:15:00 | 1724.20 | STOP_HIT | 1.00 | 1.63% |
| SELL | retest2 | 2025-10-15 12:15:00 | 1735.40 | 2025-10-15 13:15:00 | 1735.60 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-10-15 12:45:00 | 1733.70 | 2025-10-15 13:15:00 | 1735.60 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-10-20 15:15:00 | 1754.00 | 2025-11-04 10:15:00 | 1778.90 | STOP_HIT | 1.00 | 1.42% |
| BUY | retest2 | 2025-10-24 10:45:00 | 1749.80 | 2025-11-04 10:15:00 | 1778.90 | STOP_HIT | 1.00 | 1.66% |
| BUY | retest2 | 2025-10-24 11:45:00 | 1753.20 | 2025-11-04 10:15:00 | 1778.90 | STOP_HIT | 1.00 | 1.47% |
| BUY | retest2 | 2025-10-24 14:00:00 | 1750.50 | 2025-11-04 10:15:00 | 1778.90 | STOP_HIT | 1.00 | 1.62% |
| BUY | retest2 | 2025-10-27 11:15:00 | 1776.80 | 2025-11-04 10:15:00 | 1778.90 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2025-10-28 09:45:00 | 1775.50 | 2025-11-04 10:15:00 | 1778.90 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-10-30 09:30:00 | 1780.00 | 2025-11-04 10:15:00 | 1778.90 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2025-10-30 10:30:00 | 1781.90 | 2025-11-04 10:15:00 | 1778.90 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2025-10-31 09:15:00 | 1797.00 | 2025-11-04 10:15:00 | 1778.90 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-11-10 11:30:00 | 1801.50 | 2025-11-11 14:15:00 | 1790.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-11-18 09:30:00 | 1771.30 | 2025-11-26 13:15:00 | 1728.90 | STOP_HIT | 1.00 | 2.39% |
| SELL | retest2 | 2025-11-18 12:30:00 | 1768.50 | 2025-11-26 13:15:00 | 1728.90 | STOP_HIT | 1.00 | 2.24% |
| BUY | retest2 | 2025-12-26 10:15:00 | 1871.10 | 2026-01-08 09:15:00 | 1931.40 | STOP_HIT | 1.00 | 3.22% |
| BUY | retest2 | 2025-12-26 11:00:00 | 1876.60 | 2026-01-08 09:15:00 | 1931.40 | STOP_HIT | 1.00 | 2.92% |
| BUY | retest1 | 2026-01-27 09:15:00 | 2052.70 | 2026-01-29 13:15:00 | 2054.90 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2026-02-13 11:45:00 | 2257.90 | 2026-02-16 09:15:00 | 2248.70 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2026-02-16 09:15:00 | 2257.80 | 2026-02-16 09:15:00 | 2248.70 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2026-02-20 12:30:00 | 2203.10 | 2026-02-25 10:15:00 | 2222.30 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2026-02-20 14:15:00 | 2202.80 | 2026-02-25 11:15:00 | 2224.40 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2026-02-23 09:30:00 | 2206.20 | 2026-02-25 11:15:00 | 2224.40 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-02-24 09:30:00 | 2204.40 | 2026-02-25 11:15:00 | 2224.40 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-02-24 12:15:00 | 2199.10 | 2026-02-25 11:15:00 | 2224.40 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2026-03-06 14:45:00 | 2161.00 | 2026-03-09 10:15:00 | 2052.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 14:45:00 | 2161.00 | 2026-03-10 09:15:00 | 2152.10 | STOP_HIT | 0.50 | 0.41% |
| BUY | retest2 | 2026-03-27 12:15:00 | 2004.70 | 2026-03-30 09:15:00 | 1957.20 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2026-04-07 09:15:00 | 1885.00 | 2026-04-08 09:15:00 | 1983.80 | STOP_HIT | 1.00 | -5.24% |
| SELL | retest2 | 2026-04-27 11:15:00 | 2002.90 | 2026-04-30 11:15:00 | 1902.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-27 11:15:00 | 2002.90 | 2026-05-04 15:15:00 | 1877.00 | STOP_HIT | 0.50 | 6.29% |
