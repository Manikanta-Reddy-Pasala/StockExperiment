# Bombay Burmah Trading Corporation Ltd. (BBTC)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1563.30
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
| ALERT2 | 4 |
| ALERT2_SKIP | 3 |
| ALERT3 | 38 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 54 |
| PARTIAL | 11 |
| TARGET_HIT | 10 |
| STOP_HIT | 46 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 66 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 45
- **Target hits / Stop hits / Partials:** 10 / 45 / 11
- **Avg / median % per leg:** 1.05% / -1.06%
- **Sum % (uncompounded):** 69.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 0 | 0.0% | 0 | 16 | 0 | -1.78% | -28.4% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.24% | -2.2% |
| BUY @ 3rd Alert (retest2) | 15 | 0 | 0.0% | 0 | 15 | 0 | -1.75% | -26.2% |
| SELL (all) | 50 | 21 | 42.0% | 10 | 29 | 11 | 1.95% | 97.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 50 | 21 | 42.0% | 10 | 29 | 11 | 1.95% | 97.4% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.24% | -2.2% |
| retest2 (combined) | 65 | 21 | 32.3% | 10 | 44 | 11 | 1.10% | 71.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 12:15:00 | 2004.80 | 1894.81 | 1894.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 2019.00 | 1898.97 | 1896.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 1927.90 | 1962.72 | 1937.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 1927.90 | 1962.72 | 1937.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1927.90 | 1962.72 | 1937.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:00:00 | 1927.90 | 1962.72 | 1937.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 1938.70 | 1962.48 | 1937.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 13:15:00 | 1944.50 | 1961.95 | 1937.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 13:00:00 | 1941.30 | 1961.97 | 1938.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 13:30:00 | 1944.00 | 1961.79 | 1938.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 12:00:00 | 1942.10 | 1961.38 | 1938.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 1930.00 | 1961.06 | 1938.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:45:00 | 1926.90 | 1961.06 | 1938.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 1936.10 | 1960.82 | 1938.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 14:45:00 | 1943.70 | 1960.58 | 1938.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 09:30:00 | 1943.30 | 1960.00 | 1938.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 10:15:00 | 1912.50 | 1959.53 | 1938.16 | SL hit (close<static) qty=1.00 sl=1927.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-19 10:15:00 | 1912.50 | 1959.53 | 1938.16 | SL hit (close<static) qty=1.00 sl=1927.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-19 10:15:00 | 1912.50 | 1959.53 | 1938.16 | SL hit (close<static) qty=1.00 sl=1927.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-19 10:15:00 | 1912.50 | 1959.53 | 1938.16 | SL hit (close<static) qty=1.00 sl=1927.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-19 10:15:00 | 1912.50 | 1959.53 | 1938.16 | SL hit (close<static) qty=1.00 sl=1930.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-19 10:15:00 | 1912.50 | 1959.53 | 1938.16 | SL hit (close<static) qty=1.00 sl=1930.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 1942.20 | 1938.52 | 1930.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 10:00:00 | 1956.20 | 1938.70 | 1930.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 1954.70 | 1958.75 | 1943.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 14:45:00 | 1960.10 | 1958.70 | 1943.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 10:15:00 | 1961.50 | 1958.61 | 1943.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 14:00:00 | 1962.90 | 1958.78 | 1944.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 15:15:00 | 1940.10 | 1958.50 | 1944.34 | SL hit (close<static) qty=1.00 sl=1942.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-09 15:15:00 | 1940.10 | 1958.50 | 1944.34 | SL hit (close<static) qty=1.00 sl=1942.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-09 15:15:00 | 1940.10 | 1958.50 | 1944.34 | SL hit (close<static) qty=1.00 sl=1942.30 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 09:15:00 | 1963.40 | 1958.50 | 1944.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1944.10 | 1958.31 | 1944.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:00:00 | 1944.10 | 1958.31 | 1944.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 1964.10 | 1958.37 | 1944.48 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-11 15:15:00 | 1940.00 | 1958.37 | 1945.23 | SL hit (close<static) qty=1.00 sl=1942.30 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 13:30:00 | 1988.60 | 1957.87 | 1945.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 15:15:00 | 1936.10 | 1968.09 | 1953.55 | SL hit (close<static) qty=1.00 sl=1943.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 09:45:00 | 2002.00 | 1968.31 | 1953.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 2009.30 | 1970.25 | 1955.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 10:15:00 | 1935.80 | 1968.59 | 1955.47 | SL hit (close<static) qty=1.00 sl=1943.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 10:15:00 | 1935.80 | 1968.59 | 1955.47 | SL hit (close<static) qty=1.00 sl=1943.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 11:15:00 | 1923.00 | 1968.14 | 1955.31 | SL hit (close<static) qty=1.00 sl=1930.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 11:15:00 | 1923.00 | 1968.14 | 1955.31 | SL hit (close<static) qty=1.00 sl=1930.10 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 15:15:00 | 1862.60 | 1945.44 | 1945.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 11:15:00 | 1857.00 | 1942.99 | 1944.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-04 09:15:00 | 1888.10 | 1856.77 | 1888.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 1888.10 | 1856.77 | 1888.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 1888.10 | 1856.77 | 1888.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:00:00 | 1888.10 | 1856.77 | 1888.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 1873.00 | 1856.93 | 1888.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 10:45:00 | 1854.70 | 1858.06 | 1887.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 1851.10 | 1857.60 | 1885.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 14:15:00 | 1860.00 | 1857.58 | 1885.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 15:15:00 | 1855.00 | 1857.64 | 1885.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1893.00 | 1857.97 | 1884.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:30:00 | 1900.10 | 1857.97 | 1884.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 1882.00 | 1858.21 | 1884.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 11:30:00 | 1880.40 | 1858.43 | 1884.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 10:15:00 | 1880.90 | 1859.65 | 1884.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 10:45:00 | 1877.90 | 1859.85 | 1884.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 1881.90 | 1860.95 | 1884.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 1897.70 | 1861.32 | 1884.85 | SL hit (close>static) qty=1.00 sl=1893.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 1897.70 | 1861.32 | 1884.85 | SL hit (close>static) qty=1.00 sl=1893.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 1897.70 | 1861.32 | 1884.85 | SL hit (close>static) qty=1.00 sl=1893.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 1897.70 | 1861.32 | 1884.85 | SL hit (close>static) qty=1.00 sl=1893.90 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 1899.90 | 1861.70 | 1884.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:30:00 | 1899.60 | 1861.70 | 1884.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 1886.90 | 1862.13 | 1884.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:45:00 | 1887.10 | 1862.13 | 1884.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 1882.00 | 1862.33 | 1884.90 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-15 10:15:00 | 1911.10 | 1863.36 | 1884.97 | SL hit (close>static) qty=1.00 sl=1909.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 10:15:00 | 1911.10 | 1863.36 | 1884.97 | SL hit (close>static) qty=1.00 sl=1909.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 10:15:00 | 1911.10 | 1863.36 | 1884.97 | SL hit (close>static) qty=1.00 sl=1909.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 10:15:00 | 1911.10 | 1863.36 | 1884.97 | SL hit (close>static) qty=1.00 sl=1909.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 1875.00 | 1890.62 | 1895.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 12:00:00 | 1874.50 | 1890.21 | 1895.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 12:45:00 | 1876.10 | 1890.11 | 1895.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 13:15:00 | 1890.00 | 1890.11 | 1895.55 | SL hit (close>static) qty=1.00 sl=1888.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-24 13:15:00 | 1890.00 | 1890.11 | 1895.55 | SL hit (close>static) qty=1.00 sl=1888.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-24 13:15:00 | 1890.00 | 1890.11 | 1895.55 | SL hit (close>static) qty=1.00 sl=1888.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:30:00 | 1874.40 | 1890.02 | 1895.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 1780.68 | 1883.12 | 1891.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 1887.80 | 1859.88 | 1877.76 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 1887.80 | 1859.88 | 1877.76 | SL hit (close>ema200) qty=0.50 sl=1859.88 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-10-06 09:30:00 | 1913.70 | 1859.88 | 1877.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 1931.80 | 1860.59 | 1878.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:00:00 | 1931.80 | 1860.59 | 1878.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 1873.00 | 1864.39 | 1878.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 1864.70 | 1864.39 | 1878.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 10:00:00 | 1869.50 | 1864.44 | 1878.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 11:00:00 | 1864.20 | 1864.44 | 1878.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 15:15:00 | 1889.00 | 1862.55 | 1876.55 | SL hit (close>static) qty=1.00 sl=1885.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-10 15:15:00 | 1889.00 | 1862.55 | 1876.55 | SL hit (close>static) qty=1.00 sl=1885.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-10 15:15:00 | 1889.00 | 1862.55 | 1876.55 | SL hit (close>static) qty=1.00 sl=1885.90 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 1863.00 | 1862.55 | 1876.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1864.70 | 1862.57 | 1876.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 10:30:00 | 1851.10 | 1862.52 | 1876.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 14:15:00 | 1886.00 | 1863.23 | 1876.48 | SL hit (close>static) qty=1.00 sl=1885.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-13 15:15:00 | 1895.00 | 1863.55 | 1876.57 | SL hit (close>static) qty=1.00 sl=1889.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 14:15:00 | 2045.50 | 1886.96 | 1886.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 15:15:00 | 2047.90 | 1888.56 | 1887.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 15:15:00 | 1945.40 | 1947.11 | 1922.97 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-06 09:15:00 | 1961.50 | 1947.11 | 1922.97 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 1917.50 | 1946.60 | 1923.07 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 1917.50 | 1946.60 | 1923.07 | SL hit (close<ema400) qty=1.00 sl=1923.07 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-11-06 12:00:00 | 1917.50 | 1946.60 | 1923.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 1913.00 | 1946.27 | 1923.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:00:00 | 1913.00 | 1946.27 | 1923.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 1915.00 | 1945.96 | 1922.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:45:00 | 1912.70 | 1945.96 | 1922.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1875.40 | 1940.15 | 1921.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:30:00 | 1867.90 | 1940.15 | 1921.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 1868.10 | 1939.44 | 1921.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:00:00 | 1868.10 | 1939.44 | 1921.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 1920.00 | 1928.40 | 1918.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 13:45:00 | 1919.30 | 1928.40 | 1918.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 1917.50 | 1928.29 | 1918.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:00:00 | 1917.50 | 1928.29 | 1918.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 1917.00 | 1928.18 | 1918.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:15:00 | 1900.60 | 1928.18 | 1918.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1896.20 | 1927.86 | 1918.76 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 14:15:00 | 1839.80 | 1910.54 | 1910.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 10:15:00 | 1836.70 | 1904.08 | 1907.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 12:15:00 | 1914.30 | 1890.55 | 1899.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 12:15:00 | 1914.30 | 1890.55 | 1899.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 1914.30 | 1890.55 | 1899.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 13:15:00 | 1923.40 | 1890.55 | 1899.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 1903.80 | 1890.68 | 1899.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 14:15:00 | 1886.90 | 1890.68 | 1899.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 15:00:00 | 1871.30 | 1890.49 | 1899.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 15:15:00 | 1885.50 | 1879.58 | 1892.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 12:30:00 | 1887.90 | 1879.99 | 1892.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 13:15:00 | 1891.80 | 1880.10 | 1892.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 13:30:00 | 1896.40 | 1880.10 | 1892.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 1864.80 | 1879.95 | 1892.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 15:15:00 | 1854.00 | 1879.95 | 1892.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 09:30:00 | 1861.20 | 1879.62 | 1892.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 09:15:00 | 1853.30 | 1879.35 | 1891.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 1860.90 | 1878.78 | 1890.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1918.10 | 1879.17 | 1890.68 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 1918.10 | 1879.17 | 1890.68 | SL hit (close>static) qty=1.00 sl=1894.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 1918.10 | 1879.17 | 1890.68 | SL hit (close>static) qty=1.00 sl=1894.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 1918.10 | 1879.17 | 1890.68 | SL hit (close>static) qty=1.00 sl=1894.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 1918.10 | 1879.17 | 1890.68 | SL hit (close>static) qty=1.00 sl=1894.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-12-16 09:45:00 | 1931.70 | 1879.17 | 1890.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1901.50 | 1879.39 | 1890.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 11:15:00 | 1896.00 | 1879.39 | 1890.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 15:15:00 | 1890.00 | 1880.00 | 1890.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:00:00 | 1896.30 | 1880.48 | 1890.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 1924.80 | 1882.00 | 1890.65 | SL hit (close>static) qty=1.00 sl=1921.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 1924.80 | 1882.00 | 1890.65 | SL hit (close>static) qty=1.00 sl=1921.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 1924.80 | 1882.00 | 1890.65 | SL hit (close>static) qty=1.00 sl=1921.20 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 14:45:00 | 1893.40 | 1885.24 | 1891.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 1900.80 | 1885.39 | 1891.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:15:00 | 1899.10 | 1885.39 | 1891.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 1886.00 | 1885.40 | 1891.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 11:00:00 | 1882.90 | 1885.37 | 1891.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 15:00:00 | 1873.30 | 1878.57 | 1887.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 09:30:00 | 1882.10 | 1878.59 | 1887.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 1881.50 | 1879.40 | 1887.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1872.20 | 1879.32 | 1887.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 11:00:00 | 1864.60 | 1879.18 | 1887.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 15:15:00 | 1867.90 | 1878.65 | 1886.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 11:45:00 | 1870.90 | 1878.72 | 1886.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 12:30:00 | 1872.00 | 1878.66 | 1886.39 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1886.10 | 1878.47 | 1886.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:00:00 | 1886.10 | 1878.47 | 1886.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 1873.10 | 1878.41 | 1886.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 11:15:00 | 1867.90 | 1878.41 | 1886.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 1887.50 | 1878.14 | 1885.70 | SL hit (close>static) qty=1.00 sl=1886.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-07 10:15:00 | 1910.00 | 1878.46 | 1885.82 | SL hit (close>static) qty=1.00 sl=1895.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-07 10:15:00 | 1910.00 | 1878.46 | 1885.82 | SL hit (close>static) qty=1.00 sl=1895.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-07 10:15:00 | 1910.00 | 1878.46 | 1885.82 | SL hit (close>static) qty=1.00 sl=1895.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-07 10:15:00 | 1910.00 | 1878.46 | 1885.82 | SL hit (close>static) qty=1.00 sl=1895.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:15:00 | 1863.40 | 1879.91 | 1886.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1792.56 | 1874.07 | 1882.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1791.22 | 1874.07 | 1882.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1793.51 | 1874.07 | 1882.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1798.73 | 1874.07 | 1882.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1788.76 | 1874.07 | 1882.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1787.99 | 1874.07 | 1882.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1787.42 | 1874.07 | 1882.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 1777.73 | 1872.31 | 1881.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 1779.63 | 1872.31 | 1881.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 1770.23 | 1854.83 | 1870.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-20 14:15:00 | 1698.21 | 1851.45 | 1869.04 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-20 14:15:00 | 1684.17 | 1851.45 | 1869.04 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-20 14:15:00 | 1696.95 | 1851.45 | 1869.04 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-20 14:15:00 | 1699.11 | 1851.45 | 1869.04 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-20 14:15:00 | 1704.06 | 1851.45 | 1869.04 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-20 14:15:00 | 1694.61 | 1851.45 | 1869.04 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-20 14:15:00 | 1685.97 | 1851.45 | 1869.04 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-20 14:15:00 | 1693.89 | 1851.45 | 1869.04 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-20 14:15:00 | 1693.35 | 1851.45 | 1869.04 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-02-01 12:15:00 | 1677.06 | 1800.18 | 1836.08 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-16 13:15:00 | 1944.50 | 2025-06-19 10:15:00 | 1912.50 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-06-17 13:00:00 | 1941.30 | 2025-06-19 10:15:00 | 1912.50 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-06-17 13:30:00 | 1944.00 | 2025-06-19 10:15:00 | 1912.50 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-06-18 12:00:00 | 1942.10 | 2025-06-19 10:15:00 | 1912.50 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-06-18 14:45:00 | 1943.70 | 2025-06-19 10:15:00 | 1912.50 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-06-19 09:30:00 | 1943.30 | 2025-06-19 10:15:00 | 1912.50 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-06-27 09:15:00 | 1942.20 | 2025-07-09 15:15:00 | 1940.10 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-06-27 10:00:00 | 1956.20 | 2025-07-09 15:15:00 | 1940.10 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-07-08 14:45:00 | 1960.10 | 2025-07-09 15:15:00 | 1940.10 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-07-09 10:15:00 | 1961.50 | 2025-07-11 15:15:00 | 1940.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-07-09 14:00:00 | 1962.90 | 2025-07-22 15:15:00 | 1936.10 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-07-10 09:15:00 | 1963.40 | 2025-07-28 10:15:00 | 1935.80 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-07-15 13:30:00 | 1988.60 | 2025-07-28 10:15:00 | 1935.80 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2025-07-23 09:45:00 | 2002.00 | 2025-07-28 11:15:00 | 1923.00 | STOP_HIT | 1.00 | -3.95% |
| BUY | retest2 | 2025-07-24 09:15:00 | 2009.30 | 2025-07-28 11:15:00 | 1923.00 | STOP_HIT | 1.00 | -4.30% |
| SELL | retest2 | 2025-09-05 10:45:00 | 1854.70 | 2025-09-12 09:15:00 | 1897.70 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-09-09 09:15:00 | 1851.10 | 2025-09-12 09:15:00 | 1897.70 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-09-09 14:15:00 | 1860.00 | 2025-09-12 09:15:00 | 1897.70 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-09-09 15:15:00 | 1855.00 | 2025-09-12 09:15:00 | 1897.70 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-09-10 11:30:00 | 1880.40 | 2025-09-15 10:15:00 | 1911.10 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-09-11 10:15:00 | 1880.90 | 2025-09-15 10:15:00 | 1911.10 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-09-11 10:45:00 | 1877.90 | 2025-09-15 10:15:00 | 1911.10 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-09-12 09:15:00 | 1881.90 | 2025-09-15 10:15:00 | 1911.10 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-09-24 09:15:00 | 1875.00 | 2025-09-24 13:15:00 | 1890.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-09-24 12:00:00 | 1874.50 | 2025-09-24 13:15:00 | 1890.00 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-09-24 12:45:00 | 1876.10 | 2025-09-24 13:15:00 | 1890.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-09-24 14:30:00 | 1874.40 | 2025-09-26 14:15:00 | 1780.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 14:30:00 | 1874.40 | 2025-10-06 09:15:00 | 1887.80 | STOP_HIT | 0.50 | -0.71% |
| SELL | retest2 | 2025-10-08 09:15:00 | 1864.70 | 2025-10-10 15:15:00 | 1889.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-10-08 10:00:00 | 1869.50 | 2025-10-10 15:15:00 | 1889.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-10-08 11:00:00 | 1864.20 | 2025-10-10 15:15:00 | 1889.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-10-13 09:15:00 | 1863.00 | 2025-10-13 14:15:00 | 1886.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-10-13 10:30:00 | 1851.10 | 2025-10-13 15:15:00 | 1895.00 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest1 | 2025-11-06 09:15:00 | 1961.50 | 2025-11-06 11:15:00 | 1917.50 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-12-03 14:15:00 | 1886.90 | 2025-12-16 09:15:00 | 1918.10 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-12-03 15:00:00 | 1871.30 | 2025-12-16 09:15:00 | 1918.10 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-12-09 15:15:00 | 1885.50 | 2025-12-16 09:15:00 | 1918.10 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-12-10 12:30:00 | 1887.90 | 2025-12-16 09:15:00 | 1918.10 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-12-10 15:15:00 | 1854.00 | 2025-12-22 09:15:00 | 1924.80 | STOP_HIT | 1.00 | -3.82% |
| SELL | retest2 | 2025-12-11 09:30:00 | 1861.20 | 2025-12-22 09:15:00 | 1924.80 | STOP_HIT | 1.00 | -3.42% |
| SELL | retest2 | 2025-12-15 09:15:00 | 1853.30 | 2025-12-22 09:15:00 | 1924.80 | STOP_HIT | 1.00 | -3.86% |
| SELL | retest2 | 2025-12-16 09:15:00 | 1860.90 | 2026-01-07 09:15:00 | 1887.50 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-12-16 11:15:00 | 1896.00 | 2026-01-07 10:15:00 | 1910.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-12-18 15:15:00 | 1890.00 | 2026-01-07 10:15:00 | 1910.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-12-19 11:00:00 | 1896.30 | 2026-01-07 10:15:00 | 1910.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-12-23 14:45:00 | 1893.40 | 2026-01-07 10:15:00 | 1910.00 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-12-24 11:00:00 | 1882.90 | 2026-01-12 09:15:00 | 1792.56 | PARTIAL | 0.50 | 4.80% |
| SELL | retest2 | 2025-12-30 15:00:00 | 1873.30 | 2026-01-12 09:15:00 | 1791.22 | PARTIAL | 0.50 | 4.38% |
| SELL | retest2 | 2025-12-31 09:30:00 | 1882.10 | 2026-01-12 09:15:00 | 1793.51 | PARTIAL | 0.50 | 4.71% |
| SELL | retest2 | 2026-01-01 09:15:00 | 1881.50 | 2026-01-12 09:15:00 | 1798.73 | PARTIAL | 0.50 | 4.40% |
| SELL | retest2 | 2026-01-01 11:00:00 | 1864.60 | 2026-01-12 09:15:00 | 1788.76 | PARTIAL | 0.50 | 4.07% |
| SELL | retest2 | 2026-01-01 15:15:00 | 1867.90 | 2026-01-12 09:15:00 | 1787.99 | PARTIAL | 0.50 | 4.28% |
| SELL | retest2 | 2026-01-05 11:45:00 | 1870.90 | 2026-01-12 09:15:00 | 1787.42 | PARTIAL | 0.50 | 4.46% |
| SELL | retest2 | 2026-01-05 12:30:00 | 1872.00 | 2026-01-12 11:15:00 | 1777.73 | PARTIAL | 0.50 | 5.04% |
| SELL | retest2 | 2026-01-06 11:15:00 | 1867.90 | 2026-01-12 11:15:00 | 1779.63 | PARTIAL | 0.50 | 4.73% |
| SELL | retest2 | 2026-01-08 11:15:00 | 1863.40 | 2026-01-20 11:15:00 | 1770.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-24 11:00:00 | 1882.90 | 2026-01-20 14:15:00 | 1698.21 | TARGET_HIT | 0.50 | 9.81% |
| SELL | retest2 | 2025-12-30 15:00:00 | 1873.30 | 2026-01-20 14:15:00 | 1684.17 | TARGET_HIT | 0.50 | 10.10% |
| SELL | retest2 | 2025-12-31 09:30:00 | 1882.10 | 2026-01-20 14:15:00 | 1696.95 | TARGET_HIT | 0.50 | 9.84% |
| SELL | retest2 | 2026-01-01 09:15:00 | 1881.50 | 2026-01-20 14:15:00 | 1699.11 | TARGET_HIT | 0.50 | 9.69% |
| SELL | retest2 | 2026-01-01 11:00:00 | 1864.60 | 2026-01-20 14:15:00 | 1704.06 | TARGET_HIT | 0.50 | 8.61% |
| SELL | retest2 | 2026-01-01 15:15:00 | 1867.90 | 2026-01-20 14:15:00 | 1694.61 | TARGET_HIT | 0.50 | 9.28% |
| SELL | retest2 | 2026-01-05 11:45:00 | 1870.90 | 2026-01-20 14:15:00 | 1685.97 | TARGET_HIT | 0.50 | 9.88% |
| SELL | retest2 | 2026-01-05 12:30:00 | 1872.00 | 2026-01-20 14:15:00 | 1693.89 | TARGET_HIT | 0.50 | 9.51% |
| SELL | retest2 | 2026-01-06 11:15:00 | 1867.90 | 2026-01-20 14:15:00 | 1693.35 | TARGET_HIT | 0.50 | 9.34% |
| SELL | retest2 | 2026-01-08 11:15:00 | 1863.40 | 2026-02-01 12:15:00 | 1677.06 | TARGET_HIT | 0.50 | 10.00% |
