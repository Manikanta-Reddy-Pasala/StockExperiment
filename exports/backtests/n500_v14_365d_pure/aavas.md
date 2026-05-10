# Aavas Financiers Ltd. (AAVAS)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1446.00
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
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 28 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 32 |
| PARTIAL | 6 |
| TARGET_HIT | 6 |
| STOP_HIT | 30 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 38 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 29
- **Target hits / Stop hits / Partials:** 3 / 29 / 6
- **Avg / median % per leg:** -0.59% / -1.72%
- **Sum % (uncompounded):** -22.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -4.45% | -35.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -4.45% | -35.6% |
| SELL (all) | 30 | 9 | 30.0% | 3 | 21 | 6 | 0.44% | 13.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 30 | 9 | 30.0% | 3 | 21 | 6 | 0.44% | 13.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 38 | 9 | 23.7% | 3 | 29 | 6 | -0.59% | -22.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 09:15:00 | 1782.50 | 1867.47 | 1867.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 09:15:00 | 1765.10 | 1861.58 | 1864.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 14:15:00 | 1849.00 | 1843.88 | 1854.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-30 15:00:00 | 1849.00 | 1843.88 | 1854.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 1844.00 | 1843.88 | 1854.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 09:15:00 | 1807.00 | 1843.88 | 1854.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 13:15:00 | 1924.20 | 1835.41 | 1848.02 | SL hit (close>static) qty=1.00 sl=1869.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 14:00:00 | 1816.80 | 1848.62 | 1853.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 09:15:00 | 1815.00 | 1848.23 | 1853.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 11:00:00 | 1816.10 | 1847.61 | 1853.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 1846.60 | 1846.78 | 1852.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 1846.60 | 1846.78 | 1852.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 1855.00 | 1846.86 | 1852.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 1837.80 | 1846.86 | 1852.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 12:00:00 | 1839.10 | 1846.45 | 1852.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 13:15:00 | 1877.30 | 1846.76 | 1852.31 | SL hit (close>static) qty=1.00 sl=1869.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 13:15:00 | 1877.30 | 1846.76 | 1852.31 | SL hit (close>static) qty=1.00 sl=1869.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 13:15:00 | 1877.30 | 1846.76 | 1852.31 | SL hit (close>static) qty=1.00 sl=1869.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 13:15:00 | 1877.30 | 1846.76 | 1852.31 | SL hit (close>static) qty=1.00 sl=1861.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 13:15:00 | 1877.30 | 1846.76 | 1852.31 | SL hit (close>static) qty=1.00 sl=1861.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 1835.80 | 1847.31 | 1852.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:30:00 | 1844.10 | 1844.83 | 1850.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 1863.80 | 1842.30 | 1849.09 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-23 11:15:00 | 1863.80 | 1842.30 | 1849.09 | SL hit (close>static) qty=1.00 sl=1861.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 11:15:00 | 1863.80 | 1842.30 | 1849.09 | SL hit (close>static) qty=1.00 sl=1861.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-06-23 12:00:00 | 1863.80 | 1842.30 | 1849.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 1851.40 | 1842.39 | 1849.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 13:15:00 | 1843.60 | 1842.39 | 1849.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 14:15:00 | 1843.60 | 1842.47 | 1849.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 10:00:00 | 1845.60 | 1842.37 | 1848.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 12:15:00 | 1872.10 | 1843.03 | 1849.19 | SL hit (close>static) qty=1.00 sl=1867.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 12:15:00 | 1872.10 | 1843.03 | 1849.19 | SL hit (close>static) qty=1.00 sl=1867.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 12:15:00 | 1872.10 | 1843.03 | 1849.19 | SL hit (close>static) qty=1.00 sl=1867.90 alert=retest2 |

### Cycle 2 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 1967.40 | 1855.13 | 1854.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 10:15:00 | 1976.50 | 1856.34 | 1855.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 10:15:00 | 1901.90 | 1903.71 | 1882.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-07 11:00:00 | 1901.90 | 1903.71 | 1882.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 1883.00 | 1903.29 | 1882.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 13:30:00 | 1882.90 | 1903.29 | 1882.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 1882.00 | 1903.08 | 1882.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 15:00:00 | 1882.00 | 1903.08 | 1882.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 1886.00 | 1902.91 | 1882.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:15:00 | 1918.20 | 1902.91 | 1882.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1961.90 | 1903.50 | 1882.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 10:15:00 | 1981.60 | 1903.50 | 1882.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 11:15:00 | 1977.10 | 1904.18 | 1883.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 14:30:00 | 1977.10 | 1907.28 | 1885.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 12:45:00 | 1974.90 | 1909.86 | 1887.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 1891.00 | 1922.91 | 1899.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:00:00 | 1891.00 | 1922.91 | 1899.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 1899.90 | 1922.68 | 1899.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 14:15:00 | 1909.00 | 1922.39 | 1899.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 11:00:00 | 1901.90 | 1921.57 | 1899.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 09:30:00 | 1903.80 | 1921.07 | 1900.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 15:00:00 | 1901.80 | 1919.71 | 1900.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 1895.00 | 1919.47 | 1900.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 1890.50 | 1919.47 | 1900.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 1900.30 | 1919.27 | 1900.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:15:00 | 1884.70 | 1919.27 | 1900.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 1874.30 | 1918.83 | 1900.40 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 1874.30 | 1918.83 | 1900.40 | SL hit (close<static) qty=1.00 sl=1884.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 1874.30 | 1918.83 | 1900.40 | SL hit (close<static) qty=1.00 sl=1884.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 1874.30 | 1918.83 | 1900.40 | SL hit (close<static) qty=1.00 sl=1884.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 1874.30 | 1918.83 | 1900.40 | SL hit (close<static) qty=1.00 sl=1884.40 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-07-24 11:00:00 | 1874.30 | 1918.83 | 1900.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 1872.20 | 1918.36 | 1900.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:00:00 | 1872.20 | 1918.36 | 1900.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 1834.20 | 1915.78 | 1899.40 | SL hit (close<static) qty=1.00 sl=1863.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 1834.20 | 1915.78 | 1899.40 | SL hit (close<static) qty=1.00 sl=1863.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 1834.20 | 1915.78 | 1899.40 | SL hit (close<static) qty=1.00 sl=1863.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 1834.20 | 1915.78 | 1899.40 | SL hit (close<static) qty=1.00 sl=1863.10 alert=retest2 |

### Cycle 3 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 1741.10 | 1884.87 | 1885.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 1714.00 | 1877.37 | 1881.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 1649.00 | 1648.78 | 1716.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 09:45:00 | 1649.00 | 1648.78 | 1716.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 1672.00 | 1646.47 | 1688.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:30:00 | 1674.30 | 1646.47 | 1688.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1675.10 | 1632.93 | 1667.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 1675.10 | 1632.93 | 1667.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 1667.40 | 1633.27 | 1667.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:30:00 | 1671.00 | 1633.27 | 1667.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 1668.10 | 1633.62 | 1667.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 11:15:00 | 1652.60 | 1640.14 | 1669.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 12:15:00 | 1658.40 | 1640.33 | 1669.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 1654.90 | 1640.89 | 1668.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 1651.80 | 1641.89 | 1668.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 1665.70 | 1643.07 | 1668.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 15:00:00 | 1665.70 | 1643.07 | 1668.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 1651.00 | 1643.37 | 1668.06 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-31 09:15:00 | 1683.30 | 1644.79 | 1667.93 | SL hit (close>static) qty=1.00 sl=1674.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-31 09:15:00 | 1683.30 | 1644.79 | 1667.93 | SL hit (close>static) qty=1.00 sl=1674.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-31 09:15:00 | 1683.30 | 1644.79 | 1667.93 | SL hit (close>static) qty=1.00 sl=1674.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-31 09:15:00 | 1683.30 | 1644.79 | 1667.93 | SL hit (close>static) qty=1.00 sl=1674.90 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 15:00:00 | 1646.60 | 1646.53 | 1667.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 1646.20 | 1646.61 | 1667.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 10:45:00 | 1647.20 | 1646.64 | 1667.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 15:00:00 | 1638.50 | 1646.31 | 1666.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 1564.84 | 1643.19 | 1664.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 09:15:00 | 1564.27 | 1638.82 | 1661.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 09:15:00 | 1563.89 | 1638.82 | 1661.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1659.90 | 1639.03 | 1661.22 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 1659.90 | 1639.03 | 1661.22 | SL hit (close>ema200) qty=0.50 sl=1639.03 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 1659.90 | 1639.03 | 1661.22 | SL hit (close>ema200) qty=0.50 sl=1639.03 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 1659.90 | 1639.03 | 1661.22 | SL hit (close>ema200) qty=0.50 sl=1639.03 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-11-10 11:00:00 | 1659.90 | 1639.03 | 1661.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 1666.10 | 1639.30 | 1661.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:00:00 | 1666.10 | 1639.30 | 1661.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 1653.40 | 1639.44 | 1661.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:30:00 | 1688.90 | 1639.44 | 1661.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 1645.80 | 1639.50 | 1661.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 14:15:00 | 1636.90 | 1639.50 | 1661.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 12:30:00 | 1618.90 | 1639.87 | 1660.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 1683.40 | 1639.62 | 1660.14 | SL hit (close>static) qty=1.00 sl=1673.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 1683.40 | 1639.62 | 1660.14 | SL hit (close>static) qty=1.00 sl=1664.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 1683.40 | 1639.62 | 1660.14 | SL hit (close>static) qty=1.00 sl=1664.60 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 10:15:00 | 1637.30 | 1657.56 | 1666.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 11:15:00 | 1639.90 | 1657.41 | 1666.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 1664.00 | 1652.01 | 1662.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 1664.00 | 1652.01 | 1662.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 1650.00 | 1651.99 | 1662.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:45:00 | 1635.70 | 1651.71 | 1662.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 11:15:00 | 1557.90 | 1641.01 | 1655.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 12:15:00 | 1555.43 | 1640.15 | 1655.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 12:15:00 | 1553.91 | 1640.15 | 1655.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-03 09:15:00 | 1475.91 | 1621.02 | 1643.97 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-12-03 12:15:00 | 1473.57 | 1616.75 | 1641.48 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-12-03 12:15:00 | 1472.13 | 1616.75 | 1641.48 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 1377.40 | 1298.38 | 1298.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 13:15:00 | 1383.60 | 1299.22 | 1298.78 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-06-02 09:15:00 | 1807.00 | 2025-06-06 13:15:00 | 1924.20 | STOP_HIT | 1.00 | -6.49% |
| SELL | retest2 | 2025-06-12 14:00:00 | 1816.80 | 2025-06-16 13:15:00 | 1877.30 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2025-06-13 09:15:00 | 1815.00 | 2025-06-16 13:15:00 | 1877.30 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2025-06-13 11:00:00 | 1816.10 | 2025-06-16 13:15:00 | 1877.30 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2025-06-16 09:15:00 | 1837.80 | 2025-06-16 13:15:00 | 1877.30 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-06-16 12:00:00 | 1839.10 | 2025-06-16 13:15:00 | 1877.30 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-06-17 09:15:00 | 1835.80 | 2025-06-23 11:15:00 | 1863.80 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-06-19 10:30:00 | 1844.10 | 2025-06-23 11:15:00 | 1863.80 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-06-23 13:15:00 | 1843.60 | 2025-06-24 12:15:00 | 1872.10 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-06-23 14:15:00 | 1843.60 | 2025-06-24 12:15:00 | 1872.10 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-06-24 10:00:00 | 1845.60 | 2025-06-24 12:15:00 | 1872.10 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-07-08 10:15:00 | 1981.60 | 2025-07-24 10:15:00 | 1874.30 | STOP_HIT | 1.00 | -5.41% |
| BUY | retest2 | 2025-07-08 11:15:00 | 1977.10 | 2025-07-24 10:15:00 | 1874.30 | STOP_HIT | 1.00 | -5.20% |
| BUY | retest2 | 2025-07-08 14:30:00 | 1977.10 | 2025-07-24 10:15:00 | 1874.30 | STOP_HIT | 1.00 | -5.20% |
| BUY | retest2 | 2025-07-09 12:45:00 | 1974.90 | 2025-07-24 10:15:00 | 1874.30 | STOP_HIT | 1.00 | -5.09% |
| BUY | retest2 | 2025-07-18 14:15:00 | 1909.00 | 2025-07-25 09:15:00 | 1834.20 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest2 | 2025-07-21 11:00:00 | 1901.90 | 2025-07-25 09:15:00 | 1834.20 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest2 | 2025-07-23 09:30:00 | 1903.80 | 2025-07-25 09:15:00 | 1834.20 | STOP_HIT | 1.00 | -3.66% |
| BUY | retest2 | 2025-07-23 15:00:00 | 1901.80 | 2025-07-25 09:15:00 | 1834.20 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2025-10-27 11:15:00 | 1652.60 | 2025-10-31 09:15:00 | 1683.30 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-10-27 12:15:00 | 1658.40 | 2025-10-31 09:15:00 | 1683.30 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-10-28 09:15:00 | 1654.90 | 2025-10-31 09:15:00 | 1683.30 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-10-29 09:15:00 | 1651.80 | 2025-10-31 09:15:00 | 1683.30 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-11-03 15:00:00 | 1646.60 | 2025-11-07 09:15:00 | 1564.84 | PARTIAL | 0.50 | 4.97% |
| SELL | retest2 | 2025-11-04 09:15:00 | 1646.20 | 2025-11-10 09:15:00 | 1564.27 | PARTIAL | 0.50 | 4.98% |
| SELL | retest2 | 2025-11-04 10:45:00 | 1647.20 | 2025-11-10 09:15:00 | 1563.89 | PARTIAL | 0.50 | 5.06% |
| SELL | retest2 | 2025-11-03 15:00:00 | 1646.60 | 2025-11-10 10:15:00 | 1659.90 | STOP_HIT | 0.50 | -0.81% |
| SELL | retest2 | 2025-11-04 09:15:00 | 1646.20 | 2025-11-10 10:15:00 | 1659.90 | STOP_HIT | 0.50 | -0.83% |
| SELL | retest2 | 2025-11-04 10:45:00 | 1647.20 | 2025-11-10 10:15:00 | 1659.90 | STOP_HIT | 0.50 | -0.77% |
| SELL | retest2 | 2025-11-04 15:00:00 | 1638.50 | 2025-11-12 09:15:00 | 1683.40 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2025-11-10 14:15:00 | 1636.90 | 2025-11-12 09:15:00 | 1683.40 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2025-11-11 12:30:00 | 1618.90 | 2025-11-12 09:15:00 | 1683.40 | STOP_HIT | 1.00 | -3.98% |
| SELL | retest2 | 2025-11-19 10:15:00 | 1637.30 | 2025-11-28 11:15:00 | 1557.90 | PARTIAL | 0.50 | 4.85% |
| SELL | retest2 | 2025-11-19 11:15:00 | 1639.90 | 2025-11-28 12:15:00 | 1555.43 | PARTIAL | 0.50 | 5.15% |
| SELL | retest2 | 2025-11-25 09:45:00 | 1635.70 | 2025-11-28 12:15:00 | 1553.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-19 10:15:00 | 1637.30 | 2025-12-03 09:15:00 | 1475.91 | TARGET_HIT | 0.50 | 9.86% |
| SELL | retest2 | 2025-11-19 11:15:00 | 1639.90 | 2025-12-03 12:15:00 | 1473.57 | TARGET_HIT | 0.50 | 10.14% |
| SELL | retest2 | 2025-11-25 09:45:00 | 1635.70 | 2025-12-03 12:15:00 | 1472.13 | TARGET_HIT | 0.50 | 10.00% |
