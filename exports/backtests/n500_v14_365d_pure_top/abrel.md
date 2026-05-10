# Aditya Birla Real Estate Ltd. (ABREL)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1479.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 30 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 32 |
| PARTIAL | 17 |
| TARGET_HIT | 8 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 49 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 19
- **Target hits / Stop hits / Partials:** 8 / 24 / 17
- **Avg / median % per leg:** 2.86% / 2.45%
- **Sum % (uncompounded):** 140.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.50% | -6.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.50% | -6.0% |
| SELL (all) | 45 | 30 | 66.7% | 8 | 20 | 17 | 3.25% | 146.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 45 | 30 | 66.7% | 8 | 20 | 17 | 3.25% | 146.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 49 | 30 | 61.2% | 8 | 24 | 17 | 2.86% | 140.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 13:15:00 | 2190.90 | 2000.02 | 1999.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 14:15:00 | 2196.20 | 2001.98 | 2000.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 2315.70 | 2333.30 | 2235.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 11:45:00 | 2314.20 | 2333.30 | 2235.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 2223.00 | 2327.98 | 2239.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:45:00 | 2224.70 | 2327.98 | 2239.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 2220.40 | 2326.90 | 2239.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:00:00 | 2220.40 | 2326.90 | 2239.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 2240.00 | 2325.14 | 2239.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 2240.00 | 2325.14 | 2239.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 2242.90 | 2324.33 | 2239.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 2267.60 | 2324.33 | 2239.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 10:00:00 | 2249.80 | 2323.58 | 2239.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 11:15:00 | 2249.80 | 2322.82 | 2239.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 12:15:00 | 2217.50 | 2320.96 | 2239.55 | SL hit (close<static) qty=1.00 sl=2237.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-09 12:15:00 | 2217.50 | 2320.96 | 2239.55 | SL hit (close<static) qty=1.00 sl=2237.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-09 12:15:00 | 2217.50 | 2320.96 | 2239.55 | SL hit (close<static) qty=1.00 sl=2237.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 10:30:00 | 2250.00 | 2295.92 | 2236.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 2243.00 | 2294.88 | 2236.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:00:00 | 2243.00 | 2294.88 | 2236.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 2229.30 | 2294.23 | 2236.13 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-15 13:15:00 | 2229.30 | 2294.23 | 2236.13 | SL hit (close<static) qty=1.00 sl=2237.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-07-15 13:30:00 | 2225.70 | 2294.23 | 2236.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 2237.60 | 2293.67 | 2236.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:45:00 | 2227.40 | 2293.67 | 2236.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 2229.50 | 2293.03 | 2236.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:15:00 | 2220.10 | 2293.03 | 2236.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 2221.30 | 2292.31 | 2236.03 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-07-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 15:15:00 | 1953.80 | 2196.39 | 2197.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 10:15:00 | 1924.70 | 2174.70 | 2186.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 14:15:00 | 1865.90 | 1852.12 | 1940.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 14:45:00 | 1871.30 | 1852.12 | 1940.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 1913.90 | 1857.09 | 1939.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 14:00:00 | 1909.30 | 1864.55 | 1938.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 15:15:00 | 1900.00 | 1865.05 | 1938.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 14:00:00 | 1908.00 | 1867.85 | 1937.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:30:00 | 1907.40 | 1870.03 | 1936.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 10:15:00 | 1813.83 | 1866.88 | 1929.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 10:15:00 | 1812.60 | 1866.88 | 1929.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 10:15:00 | 1812.03 | 1866.88 | 1929.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 11:15:00 | 1805.00 | 1866.15 | 1928.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-09-29 11:15:00 | 1718.37 | 1851.78 | 1917.01 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-09-29 11:15:00 | 1710.00 | 1851.78 | 1917.01 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-09-29 11:15:00 | 1717.20 | 1851.78 | 1917.01 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-09-29 11:15:00 | 1716.66 | 1851.78 | 1917.01 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 1781.60 | 1708.31 | 1787.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:30:00 | 1790.50 | 1708.31 | 1787.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 1796.60 | 1709.19 | 1787.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:00:00 | 1796.60 | 1709.19 | 1787.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 1815.00 | 1710.25 | 1787.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:45:00 | 1817.20 | 1710.25 | 1787.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 1817.00 | 1711.31 | 1787.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 1821.00 | 1711.31 | 1787.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 1772.30 | 1754.10 | 1798.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 14:45:00 | 1771.50 | 1755.45 | 1798.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 10:00:00 | 1766.40 | 1755.70 | 1797.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 10:30:00 | 1768.60 | 1755.86 | 1797.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 11:45:00 | 1767.30 | 1755.96 | 1797.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 1785.90 | 1755.95 | 1794.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 13:30:00 | 1775.10 | 1756.17 | 1794.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 09:15:00 | 1802.20 | 1756.99 | 1794.17 | SL hit (close>static) qty=1.00 sl=1798.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-13 10:15:00 | 1814.30 | 1757.56 | 1794.27 | SL hit (close>static) qty=1.00 sl=1810.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-13 10:15:00 | 1814.30 | 1757.56 | 1794.27 | SL hit (close>static) qty=1.00 sl=1810.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-13 10:15:00 | 1814.30 | 1757.56 | 1794.27 | SL hit (close>static) qty=1.00 sl=1810.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-13 10:15:00 | 1814.30 | 1757.56 | 1794.27 | SL hit (close>static) qty=1.00 sl=1810.40 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 12:30:00 | 1778.60 | 1758.08 | 1794.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 10:15:00 | 1808.60 | 1756.87 | 1789.11 | SL hit (close>static) qty=1.00 sl=1798.20 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 11:15:00 | 1775.00 | 1756.87 | 1789.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 12:15:00 | 1800.90 | 1757.68 | 1789.19 | SL hit (close>static) qty=1.00 sl=1798.20 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 10:00:00 | 1780.70 | 1759.36 | 1789.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 1787.80 | 1759.64 | 1789.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:00:00 | 1787.80 | 1759.64 | 1789.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 1790.00 | 1759.94 | 1789.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 12:30:00 | 1780.00 | 1760.15 | 1789.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 10:30:00 | 1783.40 | 1757.84 | 1781.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 12:00:00 | 1759.20 | 1757.85 | 1780.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:00:00 | 1780.70 | 1757.65 | 1779.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 1776.00 | 1757.83 | 1779.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 13:30:00 | 1781.50 | 1757.83 | 1779.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 1776.80 | 1758.02 | 1779.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:00:00 | 1776.80 | 1758.02 | 1779.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 1773.00 | 1758.17 | 1779.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:15:00 | 1825.50 | 1758.17 | 1779.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1785.00 | 1758.44 | 1779.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:30:00 | 1816.40 | 1758.44 | 1779.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 1740.90 | 1758.26 | 1778.89 | EMA400 retest candle locked (from downside) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 10:15:00 | 1694.23 | 1758.26 | 1778.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 13:15:00 | 1758.90 | 1757.81 | 1778.35 | SL hit (close>ema200) qty=0.50 sl=1757.81 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 09:15:00 | 1691.66 | 1756.82 | 1777.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 09:15:00 | 1691.00 | 1756.82 | 1777.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 09:15:00 | 1671.24 | 1756.82 | 1777.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 09:15:00 | 1691.66 | 1756.82 | 1777.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:30:00 | 1686.00 | 1756.82 | 1777.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 12:15:00 | 1678.20 | 1755.48 | 1776.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 11:15:00 | 1729.00 | 1726.65 | 1757.46 | SL hit (close>ema200) qty=0.50 sl=1726.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-15 11:15:00 | 1729.00 | 1726.65 | 1757.46 | SL hit (close>ema200) qty=0.50 sl=1726.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-15 11:15:00 | 1729.00 | 1726.65 | 1757.46 | SL hit (close>ema200) qty=0.50 sl=1726.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-15 11:15:00 | 1729.00 | 1726.65 | 1757.46 | SL hit (close>ema200) qty=0.50 sl=1726.65 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 1685.10 | 1726.35 | 1756.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 11:30:00 | 1690.70 | 1725.45 | 1755.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 1741.70 | 1724.80 | 1754.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:45:00 | 1755.20 | 1724.80 | 1754.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 1753.20 | 1725.08 | 1754.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:45:00 | 1752.10 | 1725.08 | 1754.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 1743.80 | 1725.27 | 1754.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:15:00 | 1744.50 | 1725.27 | 1754.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 1743.50 | 1725.45 | 1754.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 15:15:00 | 1735.50 | 1725.87 | 1754.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 09:45:00 | 1734.00 | 1725.59 | 1753.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 09:15:00 | 1738.80 | 1727.54 | 1752.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 10:15:00 | 1736.00 | 1727.70 | 1752.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 14:15:00 | 1648.72 | 1720.69 | 1745.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 14:15:00 | 1647.30 | 1720.69 | 1745.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 14:15:00 | 1651.86 | 1720.69 | 1745.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 14:15:00 | 1649.20 | 1720.69 | 1745.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 14:15:00 | 1707.00 | 1704.55 | 1732.76 | SL hit (close>ema200) qty=0.50 sl=1704.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 14:15:00 | 1707.00 | 1704.55 | 1732.76 | SL hit (close>ema200) qty=0.50 sl=1704.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 14:15:00 | 1707.00 | 1704.55 | 1732.76 | SL hit (close>ema200) qty=0.50 sl=1704.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 14:15:00 | 1707.00 | 1704.55 | 1732.76 | SL hit (close>ema200) qty=0.50 sl=1704.55 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 1606.16 | 1695.14 | 1724.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1601.70 | 1689.86 | 1721.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1594.29 | 1689.86 | 1721.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1600.84 | 1689.86 | 1721.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-19 09:15:00 | 1517.40 | 1654.62 | 1698.25 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-19 09:15:00 | 1510.38 | 1654.62 | 1698.25 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-19 09:15:00 | 1516.59 | 1654.62 | 1698.25 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-19 09:15:00 | 1521.63 | 1654.62 | 1698.25 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1304.60 | 1203.86 | 1296.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:45:00 | 1319.30 | 1203.86 | 1296.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 1294.00 | 1204.76 | 1296.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:30:00 | 1295.70 | 1204.76 | 1296.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 1264.90 | 1205.36 | 1296.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 14:00:00 | 1257.50 | 1206.49 | 1296.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 15:15:00 | 1256.00 | 1207.06 | 1296.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:30:00 | 1254.60 | 1207.95 | 1295.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 1332.20 | 1213.21 | 1295.49 | SL hit (close>static) qty=1.00 sl=1300.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 1332.20 | 1213.21 | 1295.49 | SL hit (close>static) qty=1.00 sl=1300.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 1332.20 | 1213.21 | 1295.49 | SL hit (close>static) qty=1.00 sl=1300.40 alert=retest2 |

### Cycle 3 — BUY (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 13:15:00 | 1546.40 | 1344.76 | 1344.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 14:15:00 | 1584.80 | 1396.96 | 1372.83 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-16 09:30:00 | 2042.50 | 2025-05-16 10:15:00 | 2090.40 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-07-09 09:15:00 | 2267.60 | 2025-07-09 12:15:00 | 2217.50 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-07-09 10:00:00 | 2249.80 | 2025-07-09 12:15:00 | 2217.50 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-07-09 11:15:00 | 2249.80 | 2025-07-09 12:15:00 | 2217.50 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-07-15 10:30:00 | 2250.00 | 2025-07-15 13:15:00 | 2229.30 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-09-18 14:00:00 | 1909.30 | 2025-09-25 10:15:00 | 1813.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 15:15:00 | 1900.00 | 2025-09-25 10:15:00 | 1812.60 | PARTIAL | 0.50 | 4.60% |
| SELL | retest2 | 2025-09-19 14:00:00 | 1908.00 | 2025-09-25 10:15:00 | 1812.03 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2025-09-22 13:30:00 | 1907.40 | 2025-09-25 11:15:00 | 1805.00 | PARTIAL | 0.50 | 5.37% |
| SELL | retest2 | 2025-09-18 14:00:00 | 1909.30 | 2025-09-29 11:15:00 | 1718.37 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-18 15:15:00 | 1900.00 | 2025-09-29 11:15:00 | 1710.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-19 14:00:00 | 1908.00 | 2025-09-29 11:15:00 | 1717.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-22 13:30:00 | 1907.40 | 2025-09-29 11:15:00 | 1716.66 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-07 14:45:00 | 1771.50 | 2025-11-13 09:15:00 | 1802.20 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-11-10 10:00:00 | 1766.40 | 2025-11-13 10:15:00 | 1814.30 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-11-10 10:30:00 | 1768.60 | 2025-11-13 10:15:00 | 1814.30 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2025-11-10 11:45:00 | 1767.30 | 2025-11-13 10:15:00 | 1814.30 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-11-12 13:30:00 | 1775.10 | 2025-11-13 10:15:00 | 1814.30 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-11-13 12:30:00 | 1778.60 | 2025-11-19 10:15:00 | 1808.60 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-11-19 11:15:00 | 1775.00 | 2025-11-19 12:15:00 | 1800.90 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-11-20 10:00:00 | 1780.70 | 2025-12-05 10:15:00 | 1694.23 | PARTIAL | 0.50 | 4.86% |
| SELL | retest2 | 2025-11-20 10:00:00 | 1780.70 | 2025-12-05 13:15:00 | 1758.90 | STOP_HIT | 0.50 | 1.22% |
| SELL | retest2 | 2025-11-20 12:30:00 | 1780.00 | 2025-12-08 09:15:00 | 1691.66 | PARTIAL | 0.50 | 4.96% |
| SELL | retest2 | 2025-12-02 10:30:00 | 1783.40 | 2025-12-08 09:15:00 | 1691.00 | PARTIAL | 0.50 | 5.18% |
| SELL | retest2 | 2025-12-02 12:00:00 | 1759.20 | 2025-12-08 09:15:00 | 1671.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 13:00:00 | 1780.70 | 2025-12-08 09:15:00 | 1691.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 12:30:00 | 1780.00 | 2025-12-15 11:15:00 | 1729.00 | STOP_HIT | 0.50 | 2.87% |
| SELL | retest2 | 2025-12-02 10:30:00 | 1783.40 | 2025-12-15 11:15:00 | 1729.00 | STOP_HIT | 0.50 | 3.05% |
| SELL | retest2 | 2025-12-02 12:00:00 | 1759.20 | 2025-12-15 11:15:00 | 1729.00 | STOP_HIT | 0.50 | 1.72% |
| SELL | retest2 | 2025-12-04 13:00:00 | 1780.70 | 2025-12-15 11:15:00 | 1729.00 | STOP_HIT | 0.50 | 2.90% |
| SELL | retest2 | 2025-12-08 09:30:00 | 1686.00 | 2025-12-29 14:15:00 | 1648.72 | PARTIAL | 0.50 | 2.21% |
| SELL | retest2 | 2025-12-08 12:15:00 | 1678.20 | 2025-12-29 14:15:00 | 1647.30 | PARTIAL | 0.50 | 1.84% |
| SELL | retest2 | 2025-12-16 09:15:00 | 1685.10 | 2025-12-29 14:15:00 | 1651.86 | PARTIAL | 0.50 | 1.97% |
| SELL | retest2 | 2025-12-16 11:30:00 | 1690.70 | 2025-12-29 14:15:00 | 1649.20 | PARTIAL | 0.50 | 2.45% |
| SELL | retest2 | 2025-12-08 09:30:00 | 1686.00 | 2026-01-05 14:15:00 | 1707.00 | STOP_HIT | 0.50 | -1.25% |
| SELL | retest2 | 2025-12-08 12:15:00 | 1678.20 | 2026-01-05 14:15:00 | 1707.00 | STOP_HIT | 0.50 | -1.72% |
| SELL | retest2 | 2025-12-16 09:15:00 | 1685.10 | 2026-01-05 14:15:00 | 1707.00 | STOP_HIT | 0.50 | -1.30% |
| SELL | retest2 | 2025-12-16 11:30:00 | 1690.70 | 2026-01-05 14:15:00 | 1707.00 | STOP_HIT | 0.50 | -0.96% |
| SELL | retest2 | 2025-12-17 15:15:00 | 1735.50 | 2026-01-09 09:15:00 | 1606.16 | PARTIAL | 0.50 | 7.45% |
| SELL | retest2 | 2025-12-19 09:45:00 | 1734.00 | 2026-01-12 09:15:00 | 1601.70 | PARTIAL | 0.50 | 7.63% |
| SELL | retest2 | 2025-12-23 09:15:00 | 1738.80 | 2026-01-12 09:15:00 | 1594.29 | PARTIAL | 0.50 | 8.31% |
| SELL | retest2 | 2025-12-23 10:15:00 | 1736.00 | 2026-01-12 09:15:00 | 1600.84 | PARTIAL | 0.50 | 7.79% |
| SELL | retest2 | 2025-12-17 15:15:00 | 1735.50 | 2026-01-19 09:15:00 | 1517.40 | TARGET_HIT | 0.50 | 12.57% |
| SELL | retest2 | 2025-12-19 09:45:00 | 1734.00 | 2026-01-19 09:15:00 | 1510.38 | TARGET_HIT | 0.50 | 12.90% |
| SELL | retest2 | 2025-12-23 09:15:00 | 1738.80 | 2026-01-19 09:15:00 | 1516.59 | TARGET_HIT | 0.50 | 12.78% |
| SELL | retest2 | 2025-12-23 10:15:00 | 1736.00 | 2026-01-19 09:15:00 | 1521.63 | TARGET_HIT | 0.50 | 12.35% |
| SELL | retest2 | 2026-04-08 14:00:00 | 1257.50 | 2026-04-10 09:15:00 | 1332.20 | STOP_HIT | 1.00 | -5.94% |
| SELL | retest2 | 2026-04-08 15:15:00 | 1256.00 | 2026-04-10 09:15:00 | 1332.20 | STOP_HIT | 1.00 | -6.07% |
| SELL | retest2 | 2026-04-09 09:30:00 | 1254.60 | 2026-04-10 09:15:00 | 1332.20 | STOP_HIT | 1.00 | -6.19% |
