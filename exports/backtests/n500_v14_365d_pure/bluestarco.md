# Blue Star Ltd. (BLUESTARCO)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1691.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 38 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 35 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 32 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 34 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 4 / 30
- **Target hits / Stop hits / Partials:** 1 / 32 / 1
- **Avg / median % per leg:** -2.21% / -2.33%
- **Sum % (uncompounded):** -75.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 1 | 4.8% | 1 | 20 | 0 | -2.84% | -59.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 21 | 1 | 4.8% | 1 | 20 | 0 | -2.84% | -59.6% |
| SELL (all) | 13 | 3 | 23.1% | 0 | 12 | 1 | -1.19% | -15.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 13 | 3 | 23.1% | 0 | 12 | 1 | -1.19% | -15.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 34 | 4 | 11.8% | 1 | 32 | 1 | -2.21% | -75.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 13:15:00 | 1873.60 | 1734.88 | 1734.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 1901.10 | 1738.99 | 1736.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 09:15:00 | 1757.30 | 1759.64 | 1747.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 1757.30 | 1759.64 | 1747.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1757.30 | 1759.64 | 1747.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 13:45:00 | 1770.40 | 1758.78 | 1748.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 09:15:00 | 1779.60 | 1758.92 | 1748.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 10:15:00 | 1721.90 | 1757.30 | 1748.35 | SL hit (close<static) qty=1.00 sl=1730.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 10:15:00 | 1721.90 | 1757.30 | 1748.35 | SL hit (close<static) qty=1.00 sl=1730.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 10:00:00 | 1773.00 | 1750.99 | 1746.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 11:00:00 | 1770.00 | 1751.18 | 1746.48 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 1747.40 | 1751.26 | 1746.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 14:00:00 | 1747.40 | 1751.26 | 1746.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 1731.60 | 1751.07 | 1746.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 1731.60 | 1751.07 | 1746.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 1736.80 | 1750.93 | 1746.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 15:15:00 | 1786.30 | 1750.78 | 1746.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 09:45:00 | 1751.90 | 1758.21 | 1750.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 10:15:00 | 1727.70 | 1757.91 | 1750.86 | SL hit (close<static) qty=1.00 sl=1730.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-12 10:15:00 | 1727.70 | 1757.91 | 1750.86 | SL hit (close<static) qty=1.00 sl=1730.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 10:30:00 | 1750.00 | 1757.91 | 1750.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 11:15:00 | 1717.70 | 1757.51 | 1750.70 | SL hit (close<static) qty=1.00 sl=1726.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-12 11:15:00 | 1717.70 | 1757.51 | 1750.70 | SL hit (close<static) qty=1.00 sl=1726.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-12 11:15:00 | 1717.70 | 1757.51 | 1750.70 | SL hit (close<static) qty=1.00 sl=1726.50 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 15:15:00 | 1750.00 | 1756.97 | 1750.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 1750.50 | 1757.24 | 1750.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 1752.60 | 1757.24 | 1750.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 1761.10 | 1757.28 | 1750.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:30:00 | 1757.70 | 1757.28 | 1750.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Target hit | 2025-08-18 09:15:00 | 1925.00 | 1760.13 | 1752.62 | Target hit (10%) qty=1.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1888.00 | 1905.42 | 1862.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 10:15:00 | 1891.80 | 1905.42 | 1862.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 11:00:00 | 1894.40 | 1905.31 | 1862.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 12:15:00 | 1892.10 | 1905.14 | 1862.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 14:00:00 | 1894.70 | 1904.84 | 1862.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1855.20 | 1902.82 | 1863.79 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 1855.20 | 1902.82 | 1863.79 | SL hit (close<static) qty=1.00 sl=1861.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 1855.20 | 1902.82 | 1863.79 | SL hit (close<static) qty=1.00 sl=1861.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 1855.20 | 1902.82 | 1863.79 | SL hit (close<static) qty=1.00 sl=1861.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 1855.20 | 1902.82 | 1863.79 | SL hit (close<static) qty=1.00 sl=1861.40 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-10-01 10:00:00 | 1855.20 | 1902.82 | 1863.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 1849.90 | 1902.30 | 1863.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:45:00 | 1850.30 | 1902.30 | 1863.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 1865.10 | 1899.15 | 1863.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 13:00:00 | 1865.10 | 1899.15 | 1863.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1892.60 | 1899.08 | 1866.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 09:15:00 | 1924.80 | 1897.47 | 1868.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 13:45:00 | 1921.80 | 1900.67 | 1871.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 1937.10 | 1900.37 | 1873.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 15:00:00 | 1929.00 | 1932.93 | 1903.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 1909.90 | 1932.70 | 1903.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 09:15:00 | 1961.90 | 1932.70 | 1903.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 1817.80 | 1931.86 | 1903.14 | SL hit (close<static) qty=1.00 sl=1864.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 1817.80 | 1931.86 | 1903.14 | SL hit (close<static) qty=1.00 sl=1864.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 1817.80 | 1931.86 | 1903.14 | SL hit (close<static) qty=1.00 sl=1864.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 1817.80 | 1931.86 | 1903.14 | SL hit (close<static) qty=1.00 sl=1864.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 1817.80 | 1931.86 | 1903.14 | SL hit (close<static) qty=1.00 sl=1890.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 1776.00 | 1879.36 | 1879.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 1772.70 | 1877.32 | 1878.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 13:15:00 | 1787.00 | 1784.45 | 1817.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-12 14:00:00 | 1787.00 | 1784.45 | 1817.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 1815.00 | 1785.19 | 1816.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 1815.00 | 1785.19 | 1816.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 1809.40 | 1785.43 | 1816.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 12:30:00 | 1806.30 | 1785.67 | 1816.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 14:30:00 | 1805.60 | 1786.08 | 1816.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 15:00:00 | 1805.70 | 1786.08 | 1816.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 13:45:00 | 1805.80 | 1787.38 | 1816.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 1815.50 | 1787.66 | 1816.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 15:00:00 | 1815.50 | 1787.66 | 1816.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 1814.00 | 1787.92 | 1816.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 1814.00 | 1787.92 | 1816.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 1815.30 | 1788.20 | 1816.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:30:00 | 1815.90 | 1788.20 | 1816.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 1821.00 | 1788.52 | 1816.51 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-17 10:15:00 | 1821.00 | 1788.52 | 1816.51 | SL hit (close>static) qty=1.00 sl=1817.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 10:15:00 | 1821.00 | 1788.52 | 1816.51 | SL hit (close>static) qty=1.00 sl=1817.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 10:15:00 | 1821.00 | 1788.52 | 1816.51 | SL hit (close>static) qty=1.00 sl=1817.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 10:15:00 | 1821.00 | 1788.52 | 1816.51 | SL hit (close>static) qty=1.00 sl=1817.60 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-12-17 11:00:00 | 1821.00 | 1788.52 | 1816.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 1822.00 | 1788.86 | 1816.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:45:00 | 1824.00 | 1788.86 | 1816.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 1839.00 | 1791.09 | 1816.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:00:00 | 1839.00 | 1791.09 | 1816.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 1793.90 | 1776.68 | 1801.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:30:00 | 1792.00 | 1776.68 | 1801.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 1822.00 | 1777.13 | 1801.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:45:00 | 1822.30 | 1777.13 | 1801.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 1822.90 | 1777.59 | 1801.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:00:00 | 1822.90 | 1777.59 | 1801.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 1842.40 | 1779.93 | 1802.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:45:00 | 1830.70 | 1779.93 | 1802.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 1810.20 | 1786.77 | 1804.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:30:00 | 1811.60 | 1786.77 | 1804.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 1811.20 | 1787.14 | 1804.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:00:00 | 1811.20 | 1787.14 | 1804.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 1813.00 | 1787.40 | 1804.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:15:00 | 1822.80 | 1787.40 | 1804.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 1826.30 | 1791.57 | 1805.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 11:30:00 | 1821.20 | 1791.87 | 1805.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 12:15:00 | 1730.14 | 1791.66 | 1802.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-30 09:15:00 | 1786.50 | 1760.23 | 1783.17 | SL hit (close>ema200) qty=0.50 sl=1760.23 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 09:30:00 | 1819.90 | 1763.54 | 1784.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 10:00:00 | 1822.10 | 1763.54 | 1784.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 10:30:00 | 1820.60 | 1764.17 | 1784.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 1793.90 | 1765.45 | 1784.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 09:15:00 | 1769.50 | 1765.99 | 1784.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 13:15:00 | 1812.40 | 1767.70 | 1785.11 | SL hit (close>static) qty=1.00 sl=1811.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1883.30 | 1770.24 | 1786.12 | SL hit (close>static) qty=1.00 sl=1843.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1883.30 | 1770.24 | 1786.12 | SL hit (close>static) qty=1.00 sl=1843.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1883.30 | 1770.24 | 1786.12 | SL hit (close>static) qty=1.00 sl=1843.10 alert=retest2 |

### Cycle 3 — BUY (started 2026-02-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 15:15:00 | 1945.00 | 1800.33 | 1799.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 1954.90 | 1801.87 | 1800.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 11:15:00 | 1903.50 | 1905.01 | 1865.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-02 12:00:00 | 1903.50 | 1905.01 | 1865.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 1881.00 | 1904.47 | 1866.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 11:00:00 | 1889.10 | 1901.21 | 1866.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 12:30:00 | 1884.80 | 1904.65 | 1870.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 09:15:00 | 1900.60 | 1903.65 | 1870.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 10:00:00 | 1909.20 | 1903.71 | 1871.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 1869.50 | 1909.00 | 1877.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 09:45:00 | 1862.40 | 1909.00 | 1877.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 1880.90 | 1908.72 | 1877.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:30:00 | 1851.00 | 1908.72 | 1877.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 1864.20 | 1908.28 | 1877.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 12:00:00 | 1864.20 | 1908.28 | 1877.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 12:15:00 | 1835.60 | 1907.55 | 1876.86 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 1835.60 | 1907.55 | 1876.86 | SL hit (close<static) qty=1.00 sl=1841.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 1835.60 | 1907.55 | 1876.86 | SL hit (close<static) qty=1.00 sl=1841.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 1835.60 | 1907.55 | 1876.86 | SL hit (close<static) qty=1.00 sl=1841.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 1835.60 | 1907.55 | 1876.86 | SL hit (close<static) qty=1.00 sl=1841.50 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-03-13 13:00:00 | 1835.60 | 1907.55 | 1876.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2026-03-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 14:15:00 | 1627.00 | 1852.28 | 1852.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 15:15:00 | 1602.00 | 1849.79 | 1851.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-13 10:15:00 | 1726.50 | 1723.92 | 1776.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-13 11:00:00 | 1726.50 | 1723.92 | 1776.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1773.60 | 1724.45 | 1774.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 09:45:00 | 1784.60 | 1724.45 | 1774.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 10:15:00 | 1772.00 | 1724.92 | 1774.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 10:45:00 | 1762.00 | 1797.60 | 1801.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 12:00:00 | 1767.60 | 1797.30 | 1801.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1803.10 | 1796.71 | 1801.37 | SL hit (close>static) qty=1.00 sl=1793.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1803.10 | 1796.71 | 1801.37 | SL hit (close>static) qty=1.00 sl=1793.90 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 09:30:00 | 1761.60 | 1796.50 | 1800.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 11:15:00 | 1794.00 | 1796.30 | 1800.66 | SL hit (close>static) qty=1.00 sl=1793.90 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 11:45:00 | 1768.00 | 1796.30 | 1800.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 12:15:00 | 1752.40 | 1795.86 | 1800.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 14:15:00 | 1747.00 | 1795.46 | 1800.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-24 13:45:00 | 1770.40 | 2025-07-29 10:15:00 | 1721.90 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2025-07-25 09:15:00 | 1779.60 | 2025-07-29 10:15:00 | 1721.90 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2025-08-05 10:00:00 | 1773.00 | 2025-08-12 10:15:00 | 1727.70 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-08-05 11:00:00 | 1770.00 | 2025-08-12 10:15:00 | 1727.70 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2025-08-06 15:15:00 | 1786.30 | 2025-08-12 11:15:00 | 1717.70 | STOP_HIT | 1.00 | -3.84% |
| BUY | retest2 | 2025-08-12 09:45:00 | 1751.90 | 2025-08-12 11:15:00 | 1717.70 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-08-12 10:30:00 | 1750.00 | 2025-08-12 11:15:00 | 1717.70 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-08-12 15:15:00 | 1750.00 | 2025-08-18 09:15:00 | 1925.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-29 10:15:00 | 1891.80 | 2025-10-01 09:15:00 | 1855.20 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-09-29 11:00:00 | 1894.40 | 2025-10-01 09:15:00 | 1855.20 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-09-29 12:15:00 | 1892.10 | 2025-10-01 09:15:00 | 1855.20 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-09-29 14:00:00 | 1894.70 | 2025-10-01 09:15:00 | 1855.20 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-10-10 09:15:00 | 1924.80 | 2025-11-06 11:15:00 | 1817.80 | STOP_HIT | 1.00 | -5.56% |
| BUY | retest2 | 2025-10-13 13:45:00 | 1921.80 | 2025-11-06 11:15:00 | 1817.80 | STOP_HIT | 1.00 | -5.41% |
| BUY | retest2 | 2025-10-16 09:15:00 | 1937.10 | 2025-11-06 11:15:00 | 1817.80 | STOP_HIT | 1.00 | -6.16% |
| BUY | retest2 | 2025-11-04 15:00:00 | 1929.00 | 2025-11-06 11:15:00 | 1817.80 | STOP_HIT | 1.00 | -5.76% |
| BUY | retest2 | 2025-11-06 09:15:00 | 1961.90 | 2025-11-06 11:15:00 | 1817.80 | STOP_HIT | 1.00 | -7.34% |
| SELL | retest2 | 2025-12-15 12:30:00 | 1806.30 | 2025-12-17 10:15:00 | 1821.00 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-12-15 14:30:00 | 1805.60 | 2025-12-17 10:15:00 | 1821.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-12-15 15:00:00 | 1805.70 | 2025-12-17 10:15:00 | 1821.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-12-16 13:45:00 | 1805.80 | 2025-12-17 10:15:00 | 1821.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2026-01-09 11:30:00 | 1821.20 | 2026-01-20 12:15:00 | 1730.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 11:30:00 | 1821.20 | 2026-01-30 09:15:00 | 1786.50 | STOP_HIT | 0.50 | 1.91% |
| SELL | retest2 | 2026-02-01 09:30:00 | 1819.90 | 2026-02-02 13:15:00 | 1812.40 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2026-02-01 10:00:00 | 1822.10 | 2026-02-03 09:15:00 | 1883.30 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2026-02-01 10:30:00 | 1820.60 | 2026-02-03 09:15:00 | 1883.30 | STOP_HIT | 1.00 | -3.44% |
| SELL | retest2 | 2026-02-02 09:15:00 | 1769.50 | 2026-02-03 09:15:00 | 1883.30 | STOP_HIT | 1.00 | -6.43% |
| BUY | retest2 | 2026-03-05 11:00:00 | 1889.10 | 2026-03-13 12:15:00 | 1835.60 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2026-03-09 12:30:00 | 1884.80 | 2026-03-13 12:15:00 | 1835.60 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2026-03-10 09:15:00 | 1900.60 | 2026-03-13 12:15:00 | 1835.60 | STOP_HIT | 1.00 | -3.42% |
| BUY | retest2 | 2026-03-10 10:00:00 | 1909.20 | 2026-03-13 12:15:00 | 1835.60 | STOP_HIT | 1.00 | -3.86% |
| SELL | retest2 | 2026-04-30 10:45:00 | 1762.00 | 2026-05-04 09:15:00 | 1803.10 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2026-04-30 12:00:00 | 1767.60 | 2026-05-04 09:15:00 | 1803.10 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2026-05-07 09:30:00 | 1761.60 | 2026-05-07 11:15:00 | 1794.00 | STOP_HIT | 1.00 | -1.84% |
