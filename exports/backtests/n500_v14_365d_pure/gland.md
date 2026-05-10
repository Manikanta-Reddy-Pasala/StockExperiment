# Gland Pharma Ltd. (GLAND)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1906.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 14 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 34 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 35 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 34 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 31
- **Target hits / Stop hits / Partials:** 0 / 34 / 0
- **Avg / median % per leg:** -3.66% / -3.55%
- **Sum % (uncompounded):** -124.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 3 | 13.6% | 0 | 22 | 0 | -2.40% | -52.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 22 | 3 | 13.6% | 0 | 22 | 0 | -2.40% | -52.8% |
| SELL (all) | 12 | 0 | 0.0% | 0 | 12 | 0 | -5.98% | -71.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 0 | 0.0% | 0 | 12 | 0 | -5.98% | -71.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 34 | 3 | 8.8% | 0 | 34 | 0 | -3.66% | -124.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 15:15:00 | 1624.30 | 1520.47 | 1520.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 10:15:00 | 1626.80 | 1522.51 | 1521.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 10:15:00 | 1932.40 | 1935.96 | 1834.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-07 10:45:00 | 1933.30 | 1935.96 | 1834.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 1900.80 | 1937.42 | 1874.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:30:00 | 1909.80 | 1919.72 | 1875.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 09:15:00 | 1870.20 | 1918.46 | 1876.20 | SL hit (close<static) qty=1.00 sl=1873.60 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 10:15:00 | 1909.00 | 1915.92 | 1876.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 13:00:00 | 1908.10 | 1915.80 | 1876.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 13:45:00 | 1905.30 | 1915.75 | 1877.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 1923.40 | 1961.08 | 1917.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-26 13:00:00 | 1957.00 | 1960.47 | 1918.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 10:30:00 | 1942.50 | 1959.98 | 1918.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 15:00:00 | 2023.40 | 1959.95 | 1919.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:15:00 | 1946.70 | 1961.39 | 1925.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 1939.90 | 1961.17 | 1926.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:45:00 | 1932.60 | 1961.17 | 1926.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1935.80 | 1959.64 | 1926.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 14:30:00 | 1959.80 | 1958.59 | 1926.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:45:00 | 1946.80 | 1958.53 | 1927.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 13:15:00 | 1924.70 | 1956.77 | 1927.83 | SL hit (close<static) qty=1.00 sl=1925.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-10 13:15:00 | 1924.70 | 1956.77 | 1927.83 | SL hit (close<static) qty=1.00 sl=1925.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 14:00:00 | 1943.50 | 1947.10 | 1927.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 12:15:00 | 1924.10 | 1946.03 | 1927.28 | SL hit (close<static) qty=1.00 sl=1925.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 09:45:00 | 1947.00 | 1945.03 | 1927.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1927.70 | 1945.04 | 1927.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 1927.70 | 1945.04 | 1927.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 1925.00 | 1944.84 | 1927.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 1934.90 | 1944.84 | 1927.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 10:30:00 | 1936.80 | 1944.62 | 1927.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 13:30:00 | 1933.00 | 1944.36 | 1927.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 14:00:00 | 1933.70 | 1944.36 | 1927.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 1950.70 | 1944.24 | 1928.12 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-27 10:15:00 | 1920.50 | 1944.01 | 1928.08 | SL hit (close<static) qty=1.00 sl=1925.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 10:15:00 | 1920.50 | 1944.01 | 1928.08 | SL hit (close<static) qty=1.00 sl=1922.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 10:15:00 | 1920.50 | 1944.01 | 1928.08 | SL hit (close<static) qty=1.00 sl=1922.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 10:15:00 | 1920.50 | 1944.01 | 1928.08 | SL hit (close<static) qty=1.00 sl=1922.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 10:15:00 | 1920.50 | 1944.01 | 1928.08 | SL hit (close<static) qty=1.00 sl=1922.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:45:00 | 1994.00 | 1935.32 | 1925.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 1890.90 | 1937.48 | 1927.24 | SL hit (close<static) qty=1.00 sl=1928.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 10:15:00 | 1864.40 | 1932.51 | 1925.42 | SL hit (close<static) qty=1.00 sl=1873.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 10:15:00 | 1864.40 | 1932.51 | 1925.42 | SL hit (close<static) qty=1.00 sl=1873.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 10:15:00 | 1864.40 | 1932.51 | 1925.42 | SL hit (close<static) qty=1.00 sl=1873.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 10:15:00 | 1864.40 | 1932.51 | 1925.42 | SL hit (close<static) qty=1.00 sl=1880.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 10:15:00 | 1864.40 | 1932.51 | 1925.42 | SL hit (close<static) qty=1.00 sl=1880.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 10:15:00 | 1864.40 | 1932.51 | 1925.42 | SL hit (close<static) qty=1.00 sl=1880.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 10:15:00 | 1864.40 | 1932.51 | 1925.42 | SL hit (close<static) qty=1.00 sl=1880.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 12:15:00 | 1849.00 | 1918.94 | 1919.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 14:15:00 | 1838.10 | 1917.46 | 1918.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 12:15:00 | 1721.50 | 1718.61 | 1776.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 13:00:00 | 1721.50 | 1718.61 | 1776.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1734.80 | 1713.25 | 1763.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:15:00 | 1721.80 | 1713.25 | 1763.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:00:00 | 1722.50 | 1712.27 | 1753.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 15:15:00 | 1713.90 | 1706.47 | 1744.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 10:15:00 | 1811.90 | 1704.09 | 1739.17 | SL hit (close>static) qty=1.00 sl=1790.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-29 10:15:00 | 1811.90 | 1704.09 | 1739.17 | SL hit (close>static) qty=1.00 sl=1790.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-29 10:15:00 | 1811.90 | 1704.09 | 1739.17 | SL hit (close>static) qty=1.00 sl=1790.40 alert=retest2 |

### Cycle 3 — BUY (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 12:15:00 | 1836.40 | 1766.11 | 1766.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 13:15:00 | 1842.60 | 1766.87 | 1766.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 1773.60 | 1790.72 | 1779.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 1773.60 | 1790.72 | 1779.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 1773.60 | 1790.72 | 1779.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 1775.00 | 1790.72 | 1779.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 1781.50 | 1790.63 | 1779.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 12:15:00 | 1800.40 | 1790.68 | 1779.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 12:00:00 | 1802.00 | 1790.60 | 1779.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 14:45:00 | 1802.10 | 1790.90 | 1780.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 10:00:00 | 1800.00 | 1802.66 | 1788.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1783.60 | 1808.92 | 1793.67 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-02 11:15:00 | 1769.10 | 1808.17 | 1793.45 | SL hit (close<static) qty=1.00 sl=1770.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 11:15:00 | 1769.10 | 1808.17 | 1793.45 | SL hit (close<static) qty=1.00 sl=1770.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 11:15:00 | 1769.10 | 1808.17 | 1793.45 | SL hit (close<static) qty=1.00 sl=1770.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 11:15:00 | 1769.10 | 1808.17 | 1793.45 | SL hit (close<static) qty=1.00 sl=1770.80 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 14:45:00 | 1792.70 | 1807.25 | 1793.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 1745.00 | 1806.56 | 1793.00 | SL hit (close<static) qty=1.00 sl=1765.70 alert=retest2 |

### Cycle 4 — SELL (started 2026-03-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 14:15:00 | 1681.40 | 1780.28 | 1780.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 1669.70 | 1767.50 | 1774.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 10:15:00 | 1718.00 | 1701.33 | 1733.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-25 10:45:00 | 1719.40 | 1701.33 | 1733.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 1707.00 | 1701.88 | 1732.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 09:15:00 | 1681.70 | 1701.88 | 1732.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 09:45:00 | 1679.30 | 1701.59 | 1731.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 1673.30 | 1702.55 | 1730.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 13:00:00 | 1680.10 | 1701.76 | 1729.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 1723.80 | 1701.63 | 1728.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:45:00 | 1728.40 | 1701.63 | 1728.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 11:15:00 | 1707.20 | 1701.69 | 1728.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 13:00:00 | 1703.50 | 1701.71 | 1728.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 12:45:00 | 1702.70 | 1701.50 | 1727.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 12:15:00 | 1737.80 | 1703.06 | 1727.25 | SL hit (close>static) qty=1.00 sl=1736.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-08 12:15:00 | 1737.80 | 1703.06 | 1727.25 | SL hit (close>static) qty=1.00 sl=1736.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-17 12:15:00 | 1810.00 | 1717.74 | 1730.72 | SL hit (close>static) qty=1.00 sl=1790.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-17 12:15:00 | 1810.00 | 1717.74 | 1730.72 | SL hit (close>static) qty=1.00 sl=1790.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-17 12:15:00 | 1810.00 | 1717.74 | 1730.72 | SL hit (close>static) qty=1.00 sl=1790.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-17 12:15:00 | 1810.00 | 1717.74 | 1730.72 | SL hit (close>static) qty=1.00 sl=1790.00 alert=retest2 |

### Cycle 5 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 1793.60 | 1741.23 | 1741.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 10:15:00 | 1813.20 | 1741.95 | 1741.49 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-26 10:30:00 | 1522.00 | 2025-06-04 15:15:00 | 1624.30 | STOP_HIT | 1.00 | -6.72% |
| SELL | retest2 | 2025-05-27 09:15:00 | 1521.00 | 2025-06-04 15:15:00 | 1624.30 | STOP_HIT | 1.00 | -6.79% |
| SELL | retest2 | 2025-05-27 13:30:00 | 1520.00 | 2025-06-04 15:15:00 | 1624.30 | STOP_HIT | 1.00 | -6.86% |
| BUY | retest2 | 2025-09-05 09:30:00 | 1909.80 | 2025-09-08 09:15:00 | 1870.20 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-09-09 10:15:00 | 1909.00 | 2025-10-10 13:15:00 | 1924.70 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2025-09-09 13:00:00 | 1908.10 | 2025-10-10 13:15:00 | 1924.70 | STOP_HIT | 1.00 | 0.87% |
| BUY | retest2 | 2025-09-09 13:45:00 | 1905.30 | 2025-10-20 12:15:00 | 1924.10 | STOP_HIT | 1.00 | 0.99% |
| BUY | retest2 | 2025-09-26 13:00:00 | 1957.00 | 2025-10-27 10:15:00 | 1920.50 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-09-29 10:30:00 | 1942.50 | 2025-10-27 10:15:00 | 1920.50 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-09-29 15:00:00 | 2023.40 | 2025-10-27 10:15:00 | 1920.50 | STOP_HIT | 1.00 | -5.09% |
| BUY | retest2 | 2025-10-07 09:15:00 | 1946.70 | 2025-10-27 10:15:00 | 1920.50 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-10-08 14:30:00 | 1959.80 | 2025-10-27 10:15:00 | 1920.50 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-10-09 09:45:00 | 1946.80 | 2025-11-04 09:15:00 | 1890.90 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2025-10-17 14:00:00 | 1943.50 | 2025-11-07 10:15:00 | 1864.40 | STOP_HIT | 1.00 | -4.07% |
| BUY | retest2 | 2025-10-23 09:45:00 | 1947.00 | 2025-11-07 10:15:00 | 1864.40 | STOP_HIT | 1.00 | -4.24% |
| BUY | retest2 | 2025-10-24 09:15:00 | 1934.90 | 2025-11-07 10:15:00 | 1864.40 | STOP_HIT | 1.00 | -3.64% |
| BUY | retest2 | 2025-10-24 10:30:00 | 1936.80 | 2025-11-07 10:15:00 | 1864.40 | STOP_HIT | 1.00 | -3.74% |
| BUY | retest2 | 2025-10-24 13:30:00 | 1933.00 | 2025-11-07 10:15:00 | 1864.40 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest2 | 2025-10-24 14:00:00 | 1933.70 | 2025-11-07 10:15:00 | 1864.40 | STOP_HIT | 1.00 | -3.58% |
| BUY | retest2 | 2025-11-03 09:45:00 | 1994.00 | 2025-11-07 10:15:00 | 1864.40 | STOP_HIT | 1.00 | -6.50% |
| SELL | retest2 | 2026-01-08 10:15:00 | 1721.80 | 2026-01-29 10:15:00 | 1811.90 | STOP_HIT | 1.00 | -5.23% |
| SELL | retest2 | 2026-01-16 13:00:00 | 1722.50 | 2026-01-29 10:15:00 | 1811.90 | STOP_HIT | 1.00 | -5.19% |
| SELL | retest2 | 2026-01-22 15:15:00 | 1713.90 | 2026-01-29 10:15:00 | 1811.90 | STOP_HIT | 1.00 | -5.72% |
| BUY | retest2 | 2026-02-13 12:15:00 | 1800.40 | 2026-03-02 11:15:00 | 1769.10 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-02-16 12:00:00 | 1802.00 | 2026-03-02 11:15:00 | 1769.10 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2026-02-16 14:45:00 | 1802.10 | 2026-03-02 11:15:00 | 1769.10 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2026-02-24 10:00:00 | 1800.00 | 2026-03-02 11:15:00 | 1769.10 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2026-03-02 14:45:00 | 1792.70 | 2026-03-04 09:15:00 | 1745.00 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2026-03-30 09:15:00 | 1681.70 | 2026-04-08 12:15:00 | 1737.80 | STOP_HIT | 1.00 | -3.34% |
| SELL | retest2 | 2026-03-30 09:45:00 | 1679.30 | 2026-04-08 12:15:00 | 1737.80 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2026-04-02 09:15:00 | 1673.30 | 2026-04-17 12:15:00 | 1810.00 | STOP_HIT | 1.00 | -8.17% |
| SELL | retest2 | 2026-04-02 13:00:00 | 1680.10 | 2026-04-17 12:15:00 | 1810.00 | STOP_HIT | 1.00 | -7.73% |
| SELL | retest2 | 2026-04-06 13:00:00 | 1703.50 | 2026-04-17 12:15:00 | 1810.00 | STOP_HIT | 1.00 | -6.25% |
| SELL | retest2 | 2026-04-07 12:45:00 | 1702.70 | 2026-04-17 12:15:00 | 1810.00 | STOP_HIT | 1.00 | -6.30% |
