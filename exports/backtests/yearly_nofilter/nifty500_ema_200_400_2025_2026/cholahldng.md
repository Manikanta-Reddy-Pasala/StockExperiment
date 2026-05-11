# Cholamandalam Financial Holdings Ltd. (CHOLAHLDNG)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1785.00
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
| ALERT2 | 5 |
| ALERT2_SKIP | 5 |
| ALERT3 | 27 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 19 |
| PARTIAL | 9 |
| TARGET_HIT | 6 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 10
- **Target hits / Stop hits / Partials:** 6 / 13 / 9
- **Avg / median % per leg:** 3.35% / 5.00%
- **Sum % (uncompounded):** 93.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.26% | -1.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.26% | -1.3% |
| SELL (all) | 27 | 18 | 66.7% | 6 | 12 | 9 | 3.52% | 95.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 27 | 18 | 66.7% | 6 | 12 | 9 | 3.52% | 95.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 28 | 18 | 64.3% | 6 | 13 | 9 | 3.35% | 93.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 15:15:00 | 1847.90 | 1943.69 | 1943.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 09:15:00 | 1826.20 | 1942.52 | 1943.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1939.80 | 1928.72 | 1935.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 1939.80 | 1928.72 | 1935.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1939.80 | 1928.72 | 1935.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 1939.80 | 1928.72 | 1935.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 1954.50 | 1928.97 | 1936.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 15:00:00 | 1927.40 | 1929.52 | 1936.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 09:15:00 | 1922.60 | 1929.64 | 1936.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 09:15:00 | 1831.03 | 1925.09 | 1933.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 09:15:00 | 1826.47 | 1925.09 | 1933.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-29 09:15:00 | 1734.66 | 1892.08 | 1914.23 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 15:15:00 | 1942.00 | 1885.99 | 1885.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 09:15:00 | 1962.30 | 1891.80 | 1888.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 13:15:00 | 1900.00 | 1913.26 | 1900.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 13:15:00 | 1900.00 | 1913.26 | 1900.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 1900.00 | 1913.26 | 1900.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:00:00 | 1900.00 | 1913.26 | 1900.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 1890.80 | 1913.03 | 1900.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 1890.80 | 1913.03 | 1900.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 1882.00 | 1912.72 | 1900.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 1870.00 | 1912.72 | 1900.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1901.20 | 1912.57 | 1900.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:45:00 | 1900.00 | 1912.57 | 1900.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 1900.80 | 1912.45 | 1900.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 13:15:00 | 1908.90 | 1912.34 | 1900.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 1884.90 | 1924.51 | 1909.82 | SL hit (close<static) qty=1.00 sl=1899.90 alert=retest2 |

### Cycle 3 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 11:15:00 | 1803.10 | 1898.20 | 1898.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 13:15:00 | 1793.00 | 1896.27 | 1897.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 11:15:00 | 1907.40 | 1889.41 | 1893.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 11:15:00 | 1907.40 | 1889.41 | 1893.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 1907.40 | 1889.41 | 1893.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:45:00 | 1897.60 | 1889.41 | 1893.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 1921.80 | 1889.73 | 1893.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:00:00 | 1921.80 | 1889.73 | 1893.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 1901.40 | 1889.99 | 1893.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 15:15:00 | 1885.00 | 1889.99 | 1893.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 14:30:00 | 1887.70 | 1887.60 | 1892.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 12:00:00 | 1885.20 | 1887.73 | 1892.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 1910.80 | 1887.93 | 1892.39 | SL hit (close>static) qty=1.00 sl=1904.60 alert=retest2 |

### Cycle 4 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 1932.00 | 1896.29 | 1896.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 11:15:00 | 1946.80 | 1896.80 | 1896.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 1894.30 | 1898.48 | 1897.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 1894.30 | 1898.48 | 1897.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1894.30 | 1898.48 | 1897.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:00:00 | 1894.30 | 1898.48 | 1897.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 1911.10 | 1898.61 | 1897.32 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 10:15:00 | 1830.10 | 1895.88 | 1896.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 11:15:00 | 1820.50 | 1895.13 | 1895.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 11:15:00 | 1901.50 | 1876.74 | 1885.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 11:15:00 | 1901.50 | 1876.74 | 1885.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 1901.50 | 1876.74 | 1885.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:00:00 | 1901.50 | 1876.74 | 1885.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 1894.40 | 1876.92 | 1885.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 1894.40 | 1876.92 | 1885.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 1884.00 | 1876.73 | 1885.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:45:00 | 1884.00 | 1876.73 | 1885.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 1881.50 | 1876.78 | 1885.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 12:30:00 | 1880.70 | 1876.78 | 1885.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 1898.90 | 1877.00 | 1885.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:00:00 | 1898.90 | 1877.00 | 1885.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 1900.10 | 1877.23 | 1885.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:15:00 | 1891.20 | 1877.23 | 1885.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 1877.80 | 1877.48 | 1885.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 11:30:00 | 1876.00 | 1877.45 | 1885.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 14:15:00 | 1877.00 | 1867.46 | 1878.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 1852.60 | 1867.68 | 1878.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 1900.80 | 1868.13 | 1878.74 | SL hit (close>static) qty=1.00 sl=1891.40 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-08-18 15:00:00 | 1927.40 | 2025-08-20 09:15:00 | 1831.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-19 09:15:00 | 1922.60 | 2025-08-20 09:15:00 | 1826.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-18 15:00:00 | 1927.40 | 2025-08-29 09:15:00 | 1734.66 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-19 09:15:00 | 1922.60 | 2025-08-29 09:15:00 | 1730.34 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-16 10:45:00 | 1936.50 | 2025-09-17 09:15:00 | 1987.50 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2025-09-16 12:30:00 | 1935.80 | 2025-09-17 09:15:00 | 1987.50 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-09-30 09:15:00 | 1917.10 | 2025-10-01 10:15:00 | 1826.28 | PARTIAL | 0.50 | 4.74% |
| SELL | retest2 | 2025-09-30 13:45:00 | 1922.40 | 2025-10-01 10:15:00 | 1825.71 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2025-09-30 09:15:00 | 1917.10 | 2025-10-01 13:15:00 | 1883.40 | STOP_HIT | 0.50 | 1.76% |
| SELL | retest2 | 2025-09-30 13:45:00 | 1922.40 | 2025-10-01 13:15:00 | 1883.40 | STOP_HIT | 0.50 | 2.03% |
| SELL | retest2 | 2025-09-30 14:15:00 | 1921.80 | 2025-10-03 14:15:00 | 1821.24 | PARTIAL | 0.50 | 5.23% |
| SELL | retest2 | 2025-09-30 14:15:00 | 1921.80 | 2025-10-06 12:15:00 | 1880.00 | STOP_HIT | 0.50 | 2.18% |
| BUY | retest2 | 2025-11-10 13:15:00 | 1908.90 | 2025-11-19 09:15:00 | 1884.90 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-12-01 15:15:00 | 1885.00 | 2025-12-05 10:15:00 | 1910.80 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-12-03 14:30:00 | 1887.70 | 2025-12-05 10:15:00 | 1910.80 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-12-04 12:00:00 | 1885.20 | 2025-12-05 10:15:00 | 1910.80 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-12-08 13:45:00 | 1881.50 | 2025-12-09 11:15:00 | 1909.30 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-12-26 11:30:00 | 1876.00 | 2026-01-05 09:15:00 | 1900.80 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-01-01 14:15:00 | 1877.00 | 2026-01-05 09:15:00 | 1900.80 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2026-01-02 09:15:00 | 1852.60 | 2026-01-05 09:15:00 | 1900.80 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2026-01-07 11:00:00 | 1876.50 | 2026-01-16 09:15:00 | 1782.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 13:15:00 | 1878.40 | 2026-01-16 09:15:00 | 1784.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 13:45:00 | 1879.80 | 2026-01-16 09:15:00 | 1785.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 1878.10 | 2026-01-16 09:15:00 | 1784.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 11:00:00 | 1876.50 | 2026-01-20 12:15:00 | 1688.85 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-07 13:15:00 | 1878.40 | 2026-01-20 12:15:00 | 1690.56 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-07 13:45:00 | 1879.80 | 2026-01-20 12:15:00 | 1691.82 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 1878.10 | 2026-01-20 12:15:00 | 1690.29 | TARGET_HIT | 0.50 | 10.00% |
