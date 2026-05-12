# C.E. Info Systems Ltd. (MAPMYINDIA)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 957.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 85 |
| ALERT1 | 51 |
| ALERT2 | 51 |
| ALERT2_SKIP | 32 |
| ALERT3 | 125 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 50 |
| PARTIAL | 4 |
| TARGET_HIT | 5 |
| STOP_HIT | 46 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 55 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 38
- **Target hits / Stop hits / Partials:** 5 / 46 / 4
- **Avg / median % per leg:** 0.82% / -0.93%
- **Sum % (uncompounded):** 44.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 7 | 38.9% | 4 | 14 | 0 | 1.98% | 35.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 18 | 7 | 38.9% | 4 | 14 | 0 | 1.98% | 35.7% |
| SELL (all) | 37 | 10 | 27.0% | 1 | 32 | 4 | 0.25% | 9.3% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.29% | -1.3% |
| SELL @ 3rd Alert (retest2) | 36 | 10 | 27.8% | 1 | 31 | 4 | 0.29% | 10.6% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.29% | -1.3% |
| retest2 (combined) | 54 | 17 | 31.5% | 5 | 45 | 4 | 0.86% | 46.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 1941.30 | 1853.92 | 1844.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 2034.60 | 1910.09 | 1874.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 13:15:00 | 2049.60 | 2050.46 | 2004.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 14:00:00 | 2049.60 | 2050.46 | 2004.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 2076.90 | 2095.53 | 2076.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 2067.10 | 2095.53 | 2076.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 2074.10 | 2091.24 | 2076.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:30:00 | 2072.40 | 2091.24 | 2076.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 11:15:00 | 2078.10 | 2088.62 | 2076.53 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 15:15:00 | 2048.90 | 2070.42 | 2070.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 10:15:00 | 2044.50 | 2061.78 | 2066.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 14:15:00 | 1999.80 | 1999.06 | 2012.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 14:45:00 | 2000.00 | 1999.06 | 2012.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 1997.40 | 1996.81 | 2004.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 15:00:00 | 1997.40 | 1996.81 | 2004.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 1982.50 | 1992.38 | 2000.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 11:00:00 | 1976.10 | 1989.12 | 1998.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-02 09:15:00 | 1877.29 | 1929.67 | 1932.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 1938.90 | 1929.67 | 1932.83 | SL hit (close>static) qty=0.50 sl=1929.67 alert=retest2 |

### Cycle 3 — BUY (started 2025-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 12:15:00 | 1941.10 | 1935.32 | 1934.91 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 09:15:00 | 1915.90 | 1932.73 | 1934.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 11:15:00 | 1910.00 | 1925.35 | 1930.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 11:15:00 | 1921.40 | 1916.00 | 1921.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 11:15:00 | 1921.40 | 1916.00 | 1921.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 1921.40 | 1916.00 | 1921.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:30:00 | 1920.00 | 1916.00 | 1921.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 1926.00 | 1918.00 | 1922.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:30:00 | 1923.60 | 1918.00 | 1922.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 1921.00 | 1918.60 | 1922.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 09:15:00 | 1908.40 | 1917.73 | 1919.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 11:00:00 | 1909.90 | 1909.26 | 1912.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 12:00:00 | 1911.90 | 1909.79 | 1912.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 14:15:00 | 1929.00 | 1914.83 | 1914.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 14:15:00 | 1929.00 | 1914.83 | 1914.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 10:15:00 | 1949.80 | 1926.47 | 1921.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 1929.50 | 1933.99 | 1926.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 13:15:00 | 1929.50 | 1933.99 | 1926.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 1929.50 | 1933.99 | 1926.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 1934.00 | 1933.99 | 1926.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 1949.00 | 1936.99 | 1928.88 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 1796.00 | 1909.27 | 1917.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 10:15:00 | 1784.90 | 1884.40 | 1905.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-17 14:15:00 | 1745.00 | 1740.51 | 1759.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-17 14:45:00 | 1748.80 | 1740.51 | 1759.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 1759.30 | 1745.03 | 1758.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:30:00 | 1763.50 | 1745.03 | 1758.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 1749.20 | 1745.87 | 1757.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:30:00 | 1758.40 | 1745.87 | 1757.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 1753.70 | 1747.43 | 1757.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:45:00 | 1749.90 | 1747.43 | 1757.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1737.70 | 1743.79 | 1751.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:15:00 | 1733.40 | 1743.79 | 1751.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 09:15:00 | 1731.70 | 1741.52 | 1747.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 1730.50 | 1737.95 | 1742.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 1760.00 | 1742.33 | 1741.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 1760.00 | 1742.33 | 1741.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 11:15:00 | 1778.30 | 1760.05 | 1752.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 15:15:00 | 1763.10 | 1767.90 | 1763.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 15:15:00 | 1763.10 | 1767.90 | 1763.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 1763.10 | 1767.90 | 1763.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:15:00 | 1757.00 | 1767.90 | 1763.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1755.30 | 1765.38 | 1762.80 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 11:15:00 | 1753.10 | 1760.86 | 1761.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 12:15:00 | 1750.20 | 1758.73 | 1760.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 14:15:00 | 1763.00 | 1748.88 | 1751.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 14:15:00 | 1763.00 | 1748.88 | 1751.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 1763.00 | 1748.88 | 1751.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 14:45:00 | 1764.00 | 1748.88 | 1751.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 1753.60 | 1749.83 | 1752.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:15:00 | 1759.80 | 1749.83 | 1752.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 10:15:00 | 1765.80 | 1755.37 | 1754.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 13:15:00 | 1773.10 | 1761.61 | 1757.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 09:15:00 | 1766.20 | 1768.80 | 1762.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 10:00:00 | 1766.20 | 1768.80 | 1762.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 1758.40 | 1766.72 | 1762.14 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 15:15:00 | 1746.60 | 1758.41 | 1759.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 14:15:00 | 1736.90 | 1750.19 | 1754.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 10:15:00 | 1744.20 | 1739.00 | 1744.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 10:15:00 | 1744.20 | 1739.00 | 1744.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 1744.20 | 1739.00 | 1744.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:00:00 | 1744.20 | 1739.00 | 1744.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 1740.70 | 1739.34 | 1743.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 13:00:00 | 1732.00 | 1737.87 | 1742.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 14:15:00 | 1780.10 | 1746.98 | 1746.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 14:15:00 | 1780.10 | 1746.98 | 1746.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 14:15:00 | 1785.00 | 1765.48 | 1757.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 15:15:00 | 1775.60 | 1775.72 | 1768.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 09:15:00 | 1768.80 | 1775.72 | 1768.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1769.90 | 1774.56 | 1768.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:30:00 | 1767.90 | 1774.56 | 1768.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1768.20 | 1773.29 | 1768.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:00:00 | 1768.20 | 1773.29 | 1768.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 1787.90 | 1776.21 | 1770.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 14:45:00 | 1792.30 | 1782.14 | 1774.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 1767.20 | 1778.50 | 1774.78 | SL hit (close<static) qty=1.00 sl=1768.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 1780.10 | 1796.34 | 1797.02 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 15:15:00 | 1825.10 | 1799.38 | 1797.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 11:15:00 | 1850.00 | 1820.76 | 1809.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 10:15:00 | 1838.60 | 1839.39 | 1825.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-23 11:00:00 | 1838.60 | 1839.39 | 1825.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 1818.50 | 1832.68 | 1825.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:00:00 | 1818.50 | 1832.68 | 1825.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 1821.70 | 1830.49 | 1825.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 15:15:00 | 1808.10 | 1830.49 | 1825.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 1808.10 | 1826.01 | 1823.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 1824.30 | 1826.01 | 1823.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 1812.10 | 1821.80 | 1822.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 1812.10 | 1821.80 | 1822.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 11:15:00 | 1809.70 | 1819.38 | 1821.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 14:15:00 | 1842.30 | 1821.00 | 1821.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 14:15:00 | 1842.30 | 1821.00 | 1821.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 1842.30 | 1821.00 | 1821.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 15:00:00 | 1842.30 | 1821.00 | 1821.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 15:15:00 | 1823.70 | 1821.54 | 1821.37 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 1812.90 | 1819.81 | 1820.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 1801.00 | 1816.05 | 1818.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 12:15:00 | 1819.30 | 1813.91 | 1817.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 12:15:00 | 1819.30 | 1813.91 | 1817.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 1819.30 | 1813.91 | 1817.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 13:00:00 | 1819.30 | 1813.91 | 1817.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 1825.00 | 1816.13 | 1817.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 13:30:00 | 1825.50 | 1816.13 | 1817.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-25 14:15:00 | 1835.00 | 1819.90 | 1819.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-28 09:15:00 | 1842.10 | 1826.92 | 1822.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-29 09:15:00 | 1817.40 | 1836.70 | 1831.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 09:15:00 | 1817.40 | 1836.70 | 1831.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 1817.40 | 1836.70 | 1831.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:15:00 | 1815.70 | 1836.70 | 1831.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 1803.70 | 1830.10 | 1829.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:00:00 | 1803.70 | 1830.10 | 1829.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 11:15:00 | 1813.60 | 1826.80 | 1827.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 1799.10 | 1810.70 | 1815.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 12:15:00 | 1816.20 | 1809.29 | 1813.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 12:15:00 | 1816.20 | 1809.29 | 1813.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 1816.20 | 1809.29 | 1813.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 13:00:00 | 1816.20 | 1809.29 | 1813.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 1813.00 | 1810.03 | 1813.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 15:15:00 | 1792.10 | 1809.63 | 1813.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 09:30:00 | 1805.80 | 1804.96 | 1810.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 10:15:00 | 1782.40 | 1764.21 | 1763.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 10:15:00 | 1782.40 | 1764.21 | 1763.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 14:15:00 | 1811.10 | 1778.91 | 1772.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 13:15:00 | 1788.00 | 1793.35 | 1784.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 14:00:00 | 1788.00 | 1793.35 | 1784.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 1785.00 | 1792.01 | 1785.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 14:45:00 | 1797.20 | 1785.43 | 1784.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 15:15:00 | 1776.40 | 1783.62 | 1783.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 15:15:00 | 1776.40 | 1783.62 | 1783.86 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 10:15:00 | 1789.60 | 1784.30 | 1784.10 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 13:15:00 | 1780.00 | 1783.29 | 1783.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 15:15:00 | 1776.00 | 1781.92 | 1783.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 11:15:00 | 1785.00 | 1781.57 | 1782.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 11:15:00 | 1785.00 | 1781.57 | 1782.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 1785.00 | 1781.57 | 1782.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:00:00 | 1785.00 | 1781.57 | 1782.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 1785.90 | 1782.43 | 1782.79 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 13:15:00 | 1787.40 | 1783.43 | 1783.21 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 14:15:00 | 1780.00 | 1782.74 | 1782.92 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-08-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 15:15:00 | 1785.00 | 1783.19 | 1783.11 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 13:15:00 | 1780.10 | 1782.83 | 1783.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 15:15:00 | 1779.10 | 1781.87 | 1782.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 13:15:00 | 1642.30 | 1639.55 | 1653.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 14:00:00 | 1642.30 | 1639.55 | 1653.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 1638.50 | 1639.34 | 1651.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:45:00 | 1654.70 | 1639.34 | 1651.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1647.90 | 1641.96 | 1651.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:30:00 | 1649.90 | 1641.96 | 1651.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 1650.40 | 1643.65 | 1650.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:30:00 | 1650.10 | 1643.65 | 1650.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 11:15:00 | 1650.00 | 1644.92 | 1650.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 14:00:00 | 1644.10 | 1645.57 | 1650.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 10:45:00 | 1646.00 | 1646.83 | 1649.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 14:15:00 | 1643.40 | 1646.40 | 1648.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 13:30:00 | 1642.60 | 1647.04 | 1648.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 1648.00 | 1647.24 | 1648.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 1648.00 | 1647.24 | 1648.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 1647.00 | 1647.19 | 1647.99 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-08 09:15:00 | 1649.20 | 1648.18 | 1648.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 1649.20 | 1648.18 | 1648.08 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 13:15:00 | 1647.00 | 1647.98 | 1648.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 14:15:00 | 1640.60 | 1646.51 | 1647.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 1647.60 | 1645.49 | 1646.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 1647.60 | 1645.49 | 1646.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 1647.60 | 1645.49 | 1646.68 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 1647.50 | 1646.74 | 1646.69 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 11:15:00 | 1642.80 | 1646.41 | 1646.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 12:15:00 | 1640.00 | 1645.13 | 1646.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 14:15:00 | 1644.70 | 1644.43 | 1645.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 14:15:00 | 1644.70 | 1644.43 | 1645.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 1644.70 | 1644.43 | 1645.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:45:00 | 1646.50 | 1644.43 | 1645.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 1644.10 | 1644.37 | 1645.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 1640.00 | 1644.37 | 1645.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1635.60 | 1642.61 | 1644.61 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 14:15:00 | 1647.90 | 1643.15 | 1642.72 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 13:15:00 | 1637.00 | 1642.63 | 1642.91 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 1646.60 | 1642.89 | 1642.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 15:15:00 | 1649.90 | 1645.83 | 1644.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 12:15:00 | 1640.90 | 1646.09 | 1645.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 12:15:00 | 1640.90 | 1646.09 | 1645.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 1640.90 | 1646.09 | 1645.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 1640.90 | 1646.09 | 1645.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 1643.90 | 1645.65 | 1645.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:30:00 | 1642.40 | 1645.65 | 1645.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1654.20 | 1648.77 | 1646.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:30:00 | 1647.90 | 1648.77 | 1646.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 1647.90 | 1649.45 | 1647.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:30:00 | 1644.10 | 1649.45 | 1647.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 1647.50 | 1649.06 | 1647.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:15:00 | 1638.70 | 1649.06 | 1647.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 1638.60 | 1646.97 | 1647.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 1630.00 | 1637.60 | 1641.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 15:15:00 | 1634.70 | 1632.95 | 1636.80 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 09:15:00 | 1623.40 | 1632.95 | 1636.80 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1644.30 | 1635.22 | 1637.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-24 09:15:00 | 1644.30 | 1635.22 | 1637.49 | SL hit (close>ema400) qty=1.00 sl=1637.49 alert=retest1 |

### Cycle 35 — BUY (started 2025-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 14:15:00 | 1649.90 | 1622.70 | 1621.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 1652.80 | 1636.49 | 1630.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 12:15:00 | 1666.90 | 1669.08 | 1655.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 13:00:00 | 1666.90 | 1669.08 | 1655.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 1670.20 | 1680.01 | 1675.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 14:45:00 | 1699.90 | 1690.92 | 1683.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 09:15:00 | 1703.80 | 1692.12 | 1684.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 11:30:00 | 1700.10 | 1697.52 | 1689.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 14:00:00 | 1706.80 | 1700.14 | 1691.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-13 09:15:00 | 1869.89 | 1726.24 | 1705.87 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 10:15:00 | 1798.60 | 1833.59 | 1837.96 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 1839.50 | 1831.80 | 1831.47 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 09:15:00 | 1817.30 | 1828.90 | 1830.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 15:15:00 | 1805.20 | 1818.42 | 1823.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 12:15:00 | 1821.60 | 1818.45 | 1822.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 12:15:00 | 1821.60 | 1818.45 | 1822.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 1821.60 | 1818.45 | 1822.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 12:45:00 | 1823.40 | 1818.45 | 1822.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 1816.70 | 1818.10 | 1821.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 11:00:00 | 1813.00 | 1817.24 | 1820.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 13:00:00 | 1810.00 | 1814.25 | 1818.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 13:00:00 | 1811.50 | 1805.68 | 1810.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 15:15:00 | 1810.00 | 1806.44 | 1810.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 1810.00 | 1807.15 | 1810.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:45:00 | 1793.00 | 1804.92 | 1808.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 14:15:00 | 1829.90 | 1792.40 | 1795.04 | SL hit (close>static) qty=1.00 sl=1822.70 alert=retest2 |

### Cycle 39 — BUY (started 2025-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 15:15:00 | 1831.50 | 1800.22 | 1798.35 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 1788.90 | 1796.03 | 1796.65 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 14:15:00 | 1821.10 | 1797.56 | 1796.69 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 1791.40 | 1799.03 | 1799.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 11:15:00 | 1787.60 | 1796.74 | 1798.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 1768.60 | 1765.56 | 1775.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 11:45:00 | 1767.80 | 1765.56 | 1775.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 1780.20 | 1768.49 | 1775.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:45:00 | 1777.70 | 1768.49 | 1775.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 1803.90 | 1775.57 | 1778.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 1802.00 | 1775.57 | 1778.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 14:15:00 | 1816.00 | 1783.65 | 1781.91 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 1748.80 | 1784.67 | 1788.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 09:15:00 | 1724.00 | 1748.38 | 1765.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 10:15:00 | 1697.80 | 1696.06 | 1708.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-17 10:45:00 | 1696.00 | 1696.06 | 1708.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 1699.80 | 1697.42 | 1705.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:30:00 | 1713.90 | 1697.42 | 1705.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1693.70 | 1697.09 | 1703.60 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 1702.60 | 1699.06 | 1698.60 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 14:15:00 | 1679.00 | 1696.19 | 1698.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 15:15:00 | 1657.20 | 1668.35 | 1679.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 1671.00 | 1668.88 | 1678.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 1671.00 | 1668.88 | 1678.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1671.00 | 1668.88 | 1678.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:30:00 | 1673.20 | 1668.88 | 1678.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 1668.50 | 1664.60 | 1671.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 10:15:00 | 1664.10 | 1664.60 | 1671.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 11:15:00 | 1662.00 | 1665.42 | 1670.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 12:15:00 | 1664.10 | 1665.53 | 1670.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:00:00 | 1661.40 | 1664.71 | 1669.64 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 1656.60 | 1661.53 | 1667.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 15:00:00 | 1656.60 | 1661.53 | 1667.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1673.90 | 1664.56 | 1667.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:00:00 | 1673.90 | 1664.56 | 1667.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 1669.20 | 1665.49 | 1667.81 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-28 11:15:00 | 1687.70 | 1669.93 | 1669.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 11:15:00 | 1687.70 | 1669.93 | 1669.61 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 11:15:00 | 1659.90 | 1669.04 | 1670.20 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 13:15:00 | 1697.60 | 1671.61 | 1669.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 14:15:00 | 1729.40 | 1683.17 | 1675.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 09:15:00 | 1685.30 | 1705.36 | 1696.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 1685.30 | 1705.36 | 1696.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 1685.30 | 1705.36 | 1696.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:00:00 | 1685.30 | 1705.36 | 1696.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1685.70 | 1701.43 | 1695.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:30:00 | 1684.40 | 1701.43 | 1695.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 1686.50 | 1696.44 | 1693.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 13:00:00 | 1686.50 | 1696.44 | 1693.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 1688.00 | 1694.75 | 1693.32 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 14:15:00 | 1680.40 | 1691.88 | 1692.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 15:15:00 | 1676.90 | 1688.88 | 1690.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 1654.80 | 1645.33 | 1658.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 11:15:00 | 1654.80 | 1645.33 | 1658.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 1654.80 | 1645.33 | 1658.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 1654.80 | 1645.33 | 1658.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 1651.70 | 1646.60 | 1657.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 09:15:00 | 1641.90 | 1649.15 | 1656.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 15:15:00 | 1644.00 | 1635.74 | 1634.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 15:15:00 | 1644.00 | 1635.74 | 1634.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 09:15:00 | 1666.10 | 1641.81 | 1637.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 15:15:00 | 1657.00 | 1657.75 | 1649.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:15:00 | 1645.20 | 1657.75 | 1649.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1646.90 | 1655.58 | 1649.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 13:15:00 | 1650.20 | 1651.09 | 1648.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 15:15:00 | 1633.20 | 1647.05 | 1647.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 1633.20 | 1647.05 | 1647.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 1628.70 | 1637.16 | 1641.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 1648.30 | 1638.43 | 1641.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 1648.30 | 1638.43 | 1641.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 1648.30 | 1638.43 | 1641.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 1648.30 | 1638.43 | 1641.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 1659.40 | 1642.63 | 1642.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:00:00 | 1659.40 | 1642.63 | 1642.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2025-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 13:15:00 | 1656.90 | 1645.48 | 1644.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 1675.00 | 1658.07 | 1651.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 15:15:00 | 1651.30 | 1658.75 | 1653.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 15:15:00 | 1651.30 | 1658.75 | 1653.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 1651.30 | 1658.75 | 1653.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 09:30:00 | 1674.10 | 1661.88 | 1655.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 11:15:00 | 1692.00 | 1701.71 | 1702.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 11:15:00 | 1692.00 | 1701.71 | 1702.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 14:15:00 | 1677.00 | 1692.50 | 1697.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 13:15:00 | 1719.00 | 1674.47 | 1683.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 13:15:00 | 1719.00 | 1674.47 | 1683.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 1719.00 | 1674.47 | 1683.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 1719.00 | 1674.47 | 1683.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 1720.00 | 1683.58 | 1687.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:45:00 | 1721.20 | 1683.58 | 1687.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 15:15:00 | 1717.00 | 1690.26 | 1689.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 09:15:00 | 1723.00 | 1696.81 | 1692.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 14:15:00 | 1698.00 | 1719.13 | 1707.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 14:15:00 | 1698.00 | 1719.13 | 1707.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 1698.00 | 1719.13 | 1707.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 1698.00 | 1719.13 | 1707.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1687.00 | 1712.71 | 1705.76 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 12:15:00 | 1697.00 | 1701.58 | 1701.82 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 1712.00 | 1703.66 | 1702.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 14:15:00 | 1728.40 | 1708.61 | 1705.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 10:15:00 | 1709.10 | 1709.79 | 1706.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 11:00:00 | 1709.10 | 1709.79 | 1706.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 1708.30 | 1709.49 | 1706.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:30:00 | 1705.50 | 1709.49 | 1706.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 1715.20 | 1710.63 | 1707.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:30:00 | 1710.10 | 1710.63 | 1707.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 13:15:00 | 1710.70 | 1710.65 | 1707.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 13:30:00 | 1707.10 | 1710.65 | 1707.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 1740.00 | 1716.52 | 1710.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 15:15:00 | 1715.40 | 1716.52 | 1710.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 1715.40 | 1716.29 | 1711.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:30:00 | 1715.60 | 1717.38 | 1712.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 1712.30 | 1716.36 | 1712.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:00:00 | 1712.30 | 1716.36 | 1712.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 1697.20 | 1712.53 | 1710.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:45:00 | 1698.00 | 1712.53 | 1710.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 12:15:00 | 1695.40 | 1709.10 | 1709.39 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 12:15:00 | 1722.00 | 1708.48 | 1707.96 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 1701.00 | 1708.57 | 1708.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 1697.10 | 1704.89 | 1707.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 13:15:00 | 1702.10 | 1699.52 | 1702.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 13:15:00 | 1702.10 | 1699.52 | 1702.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 1702.10 | 1699.52 | 1702.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:00:00 | 1702.10 | 1699.52 | 1702.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 1697.60 | 1699.14 | 1701.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 1682.80 | 1698.51 | 1701.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 12:15:00 | 1598.66 | 1632.94 | 1658.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 14:15:00 | 1563.90 | 1563.43 | 1584.74 | SL hit (close>ema200) qty=0.50 sl=1563.43 alert=retest2 |

### Cycle 61 — BUY (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 11:15:00 | 1331.10 | 1310.39 | 1307.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 1334.80 | 1319.69 | 1313.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 1315.60 | 1322.70 | 1316.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 1315.60 | 1322.70 | 1316.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1315.60 | 1322.70 | 1316.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 1315.60 | 1322.70 | 1316.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1285.00 | 1315.16 | 1313.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1287.70 | 1315.16 | 1313.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 1283.90 | 1308.91 | 1310.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 1278.50 | 1302.83 | 1307.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1272.10 | 1259.80 | 1276.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 1272.10 | 1259.80 | 1276.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1272.10 | 1259.80 | 1276.40 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 1320.20 | 1289.76 | 1285.68 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 13:15:00 | 1272.90 | 1282.22 | 1283.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 1242.70 | 1263.36 | 1271.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 13:15:00 | 1254.30 | 1254.19 | 1263.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 13:45:00 | 1254.50 | 1254.19 | 1263.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 1256.00 | 1255.29 | 1262.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 1286.80 | 1255.29 | 1262.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 1267.80 | 1257.79 | 1263.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:30:00 | 1269.30 | 1257.79 | 1263.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 1281.80 | 1268.93 | 1267.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 1313.30 | 1283.42 | 1275.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 13:15:00 | 1292.50 | 1293.65 | 1283.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 13:45:00 | 1291.10 | 1293.65 | 1283.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1282.90 | 1292.09 | 1285.45 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 14:15:00 | 1274.10 | 1281.77 | 1282.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 1248.00 | 1273.93 | 1278.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 14:15:00 | 1284.70 | 1263.64 | 1270.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 14:15:00 | 1284.70 | 1263.64 | 1270.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 1284.70 | 1263.64 | 1270.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 15:00:00 | 1284.70 | 1263.64 | 1270.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 1270.20 | 1264.96 | 1270.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 1243.20 | 1264.96 | 1270.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 1181.04 | 1229.33 | 1247.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 15:15:00 | 1163.00 | 1161.83 | 1184.08 | SL hit (close>ema200) qty=0.50 sl=1161.83 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 1005.90 | 999.37 | 998.89 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 994.80 | 998.49 | 998.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 15:15:00 | 988.30 | 996.45 | 997.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 988.20 | 985.92 | 991.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 12:00:00 | 988.20 | 985.92 | 991.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 979.40 | 984.61 | 990.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 13:45:00 | 976.50 | 983.05 | 989.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 927.67 | 937.25 | 957.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-17 09:15:00 | 878.85 | 900.25 | 924.76 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 69 — BUY (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 13:15:00 | 918.20 | 889.10 | 886.03 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-03-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 11:15:00 | 871.30 | 886.57 | 886.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 864.40 | 882.13 | 884.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 13:15:00 | 883.80 | 882.47 | 884.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 13:15:00 | 883.80 | 882.47 | 884.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 13:15:00 | 883.80 | 882.47 | 884.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-23 13:45:00 | 884.80 | 882.47 | 884.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 14:15:00 | 864.00 | 878.77 | 882.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:00:00 | 856.00 | 871.45 | 878.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 852.00 | 867.60 | 876.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 891.70 | 873.64 | 875.36 | SL hit (close>static) qty=1.00 sl=885.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 895.00 | 877.91 | 877.14 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 859.10 | 876.90 | 878.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 13:15:00 | 853.40 | 866.85 | 872.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 862.70 | 830.79 | 844.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 862.70 | 830.79 | 844.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 862.70 | 830.79 | 844.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 862.70 | 830.79 | 844.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 860.80 | 836.79 | 845.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:30:00 | 870.10 | 836.79 | 845.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 864.10 | 851.25 | 850.85 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 840.60 | 849.86 | 850.65 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 12:15:00 | 865.70 | 853.53 | 852.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 13:15:00 | 879.10 | 858.64 | 854.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 09:15:00 | 855.55 | 862.50 | 857.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 09:15:00 | 855.55 | 862.50 | 857.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 855.55 | 862.50 | 857.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:30:00 | 857.70 | 862.50 | 857.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 865.95 | 863.19 | 858.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 12:45:00 | 878.05 | 867.54 | 861.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 15:00:00 | 878.70 | 870.52 | 863.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 10:15:00 | 923.60 | 931.36 | 931.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 10:15:00 | 923.60 | 931.36 | 931.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-16 11:15:00 | 914.90 | 928.07 | 929.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-16 14:15:00 | 923.20 | 922.83 | 926.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-16 14:45:00 | 921.85 | 922.83 | 926.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 933.85 | 925.22 | 927.13 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 11:15:00 | 945.65 | 930.94 | 929.50 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 922.90 | 932.55 | 933.68 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 10:15:00 | 937.70 | 934.47 | 934.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 11:15:00 | 943.25 | 936.23 | 935.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 13:15:00 | 930.90 | 935.78 | 935.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 13:15:00 | 930.90 | 935.78 | 935.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 930.90 | 935.78 | 935.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 14:00:00 | 930.90 | 935.78 | 935.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 932.15 | 935.06 | 934.95 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 932.05 | 934.53 | 934.73 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 11:15:00 | 937.20 | 935.22 | 935.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 12:15:00 | 956.35 | 939.45 | 936.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 14:15:00 | 948.95 | 954.03 | 948.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 14:15:00 | 948.95 | 954.03 | 948.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 948.95 | 954.03 | 948.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 14:45:00 | 948.40 | 954.03 | 948.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 952.00 | 953.63 | 949.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:15:00 | 935.00 | 953.63 | 949.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 933.45 | 949.59 | 947.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:45:00 | 926.45 | 949.59 | 947.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 922.00 | 944.07 | 945.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 918.15 | 935.81 | 941.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 953.95 | 933.85 | 937.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 953.95 | 933.85 | 937.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 953.95 | 933.85 | 937.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 951.10 | 933.85 | 937.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 954.15 | 937.91 | 939.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 958.35 | 937.91 | 939.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 948.25 | 941.66 | 940.93 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 13:15:00 | 927.00 | 938.51 | 940.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 915.50 | 931.13 | 934.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 937.60 | 927.51 | 930.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 937.60 | 927.51 | 930.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 937.60 | 927.51 | 930.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 937.60 | 927.51 | 930.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 945.80 | 931.17 | 931.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:00:00 | 945.80 | 931.17 | 931.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 935.40 | 932.01 | 931.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 961.00 | 942.36 | 938.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 12:15:00 | 947.30 | 947.40 | 941.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-06 13:00:00 | 947.30 | 947.40 | 941.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 959.45 | 963.05 | 955.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 960.85 | 963.05 | 955.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 958.05 | 961.27 | 957.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:00:00 | 958.05 | 961.27 | 957.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 957.00 | 960.42 | 957.58 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-26 11:00:00 | 1976.10 | 2025-06-02 09:15:00 | 1877.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-26 11:00:00 | 1976.10 | 2025-06-02 09:15:00 | 1938.90 | STOP_HIT | 0.50 | 1.88% |
| SELL | retest2 | 2025-06-06 09:15:00 | 1908.40 | 2025-06-09 14:15:00 | 1929.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-06-09 11:00:00 | 1909.90 | 2025-06-09 14:15:00 | 1929.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-06-09 12:00:00 | 1911.90 | 2025-06-09 14:15:00 | 1929.00 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-06-19 10:15:00 | 1733.40 | 2025-06-24 09:15:00 | 1760.00 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-06-20 09:15:00 | 1731.70 | 2025-06-24 09:15:00 | 1760.00 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-06-23 09:15:00 | 1730.50 | 2025-06-24 09:15:00 | 1760.00 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-07-07 13:00:00 | 1732.00 | 2025-07-07 14:15:00 | 1780.10 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2025-07-10 14:45:00 | 1792.30 | 2025-07-11 10:15:00 | 1767.20 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-07-11 15:00:00 | 1798.00 | 2025-07-18 09:15:00 | 1780.10 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-07-14 14:45:00 | 1799.70 | 2025-07-18 09:15:00 | 1780.10 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-07-15 10:15:00 | 1796.00 | 2025-07-18 09:15:00 | 1780.10 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-07-15 15:00:00 | 1807.00 | 2025-07-18 09:15:00 | 1780.10 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-07-24 09:15:00 | 1824.30 | 2025-07-24 10:15:00 | 1812.10 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-07-31 15:15:00 | 1792.10 | 2025-08-08 10:15:00 | 1782.40 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest2 | 2025-08-01 09:30:00 | 1805.80 | 2025-08-08 10:15:00 | 1782.40 | STOP_HIT | 1.00 | 1.30% |
| BUY | retest2 | 2025-08-13 14:45:00 | 1797.20 | 2025-08-13 15:15:00 | 1776.40 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-09-02 14:00:00 | 1644.10 | 2025-09-08 09:15:00 | 1649.20 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-09-03 10:45:00 | 1646.00 | 2025-09-08 09:15:00 | 1649.20 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-09-03 14:15:00 | 1643.40 | 2025-09-08 09:15:00 | 1649.20 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-09-04 13:30:00 | 1642.60 | 2025-09-08 09:15:00 | 1649.20 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-09-24 09:15:00 | 1623.40 | 2025-09-24 09:15:00 | 1644.30 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-09-24 11:30:00 | 1638.00 | 2025-09-30 14:15:00 | 1649.90 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-09-25 10:45:00 | 1637.30 | 2025-09-30 14:15:00 | 1649.90 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-09-29 15:15:00 | 1615.00 | 2025-09-30 14:15:00 | 1649.90 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-10-09 14:45:00 | 1699.90 | 2025-10-13 09:15:00 | 1869.89 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-10 09:15:00 | 1703.80 | 2025-10-13 09:15:00 | 1874.18 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-10 11:30:00 | 1700.10 | 2025-10-13 09:15:00 | 1870.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-10 14:00:00 | 1706.80 | 2025-10-13 09:15:00 | 1877.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-16 12:45:00 | 1845.80 | 2025-10-17 09:15:00 | 1804.00 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-10-16 14:15:00 | 1845.30 | 2025-10-17 09:15:00 | 1804.00 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-10-16 14:45:00 | 1849.60 | 2025-10-17 09:15:00 | 1804.00 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-10-27 11:00:00 | 1813.00 | 2025-10-30 14:15:00 | 1829.90 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-10-27 13:00:00 | 1810.00 | 2025-10-30 14:15:00 | 1829.90 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-10-28 13:00:00 | 1811.50 | 2025-10-30 14:15:00 | 1829.90 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-10-28 15:15:00 | 1810.00 | 2025-10-30 14:15:00 | 1829.90 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-10-29 09:45:00 | 1793.00 | 2025-10-30 14:15:00 | 1829.90 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-11-27 10:15:00 | 1664.10 | 2025-11-28 11:15:00 | 1687.70 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-11-27 11:15:00 | 1662.00 | 2025-11-28 11:15:00 | 1687.70 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-11-27 12:15:00 | 1664.10 | 2025-11-28 11:15:00 | 1687.70 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-11-27 13:00:00 | 1661.40 | 2025-11-28 11:15:00 | 1687.70 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-12-10 09:15:00 | 1641.90 | 2025-12-12 15:15:00 | 1644.00 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-12-16 13:15:00 | 1650.20 | 2025-12-16 15:15:00 | 1633.20 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-12-22 09:30:00 | 1674.10 | 2025-12-26 11:15:00 | 1692.00 | STOP_HIT | 1.00 | 1.07% |
| SELL | retest2 | 2026-01-08 09:15:00 | 1682.80 | 2026-01-09 12:15:00 | 1598.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 1682.80 | 2026-01-13 14:15:00 | 1563.90 | STOP_HIT | 0.50 | 7.07% |
| SELL | retest2 | 2026-02-13 09:15:00 | 1243.20 | 2026-02-16 09:15:00 | 1181.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-13 09:15:00 | 1243.20 | 2026-02-17 15:15:00 | 1163.00 | STOP_HIT | 0.50 | 6.45% |
| SELL | retest2 | 2026-03-12 13:45:00 | 976.50 | 2026-03-16 09:15:00 | 927.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 13:45:00 | 976.50 | 2026-03-17 09:15:00 | 878.85 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-24 10:00:00 | 856.00 | 2026-03-25 09:15:00 | 891.70 | STOP_HIT | 1.00 | -4.17% |
| SELL | retest2 | 2026-03-24 10:30:00 | 852.00 | 2026-03-25 09:15:00 | 891.70 | STOP_HIT | 1.00 | -4.66% |
| BUY | retest2 | 2026-04-06 12:45:00 | 878.05 | 2026-04-16 10:15:00 | 923.60 | STOP_HIT | 1.00 | 5.19% |
| BUY | retest2 | 2026-04-06 15:00:00 | 878.70 | 2026-04-16 10:15:00 | 923.60 | STOP_HIT | 1.00 | 5.11% |
