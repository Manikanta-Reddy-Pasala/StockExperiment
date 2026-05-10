# Glenmark Pharmaceuticals Ltd. (GLENMARK)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 2361.20
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
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 28 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 28 |
| PARTIAL | 0 |
| TARGET_HIT | 8 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 20
- **Target hits / Stop hits / Partials:** 8 / 20 / 0
- **Avg / median % per leg:** 1.46% / -1.33%
- **Sum % (uncompounded):** 40.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 8 | 32.0% | 8 | 17 | 0 | 1.91% | 47.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 25 | 8 | 32.0% | 8 | 17 | 0 | 1.91% | 47.7% |
| SELL (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.25% | -6.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.25% | -6.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 28 | 8 | 28.6% | 8 | 20 | 0 | 1.46% | 40.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 14:15:00 | 1520.00 | 1431.16 | 1431.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 1558.10 | 1433.28 | 1432.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 09:15:00 | 2022.20 | 2024.63 | 1900.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 1918.90 | 2010.03 | 1905.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1918.90 | 2010.03 | 1905.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 1902.40 | 2010.03 | 1905.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 1919.70 | 1988.19 | 1912.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 09:15:00 | 1931.20 | 1988.19 | 1912.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 11:15:00 | 1905.60 | 1986.12 | 1912.82 | SL hit (close<static) qty=1.00 sl=1910.70 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 14:30:00 | 1926.00 | 1984.22 | 1912.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 15:00:00 | 1927.20 | 1984.22 | 1912.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 09:30:00 | 1928.40 | 1983.15 | 1913.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 1918.00 | 1981.05 | 1913.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 1918.00 | 1981.05 | 1913.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 1916.90 | 1980.41 | 1913.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:45:00 | 1914.50 | 1980.41 | 1913.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Target hit | 2025-09-09 13:15:00 | 2118.60 | 1996.40 | 1932.29 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-09-10 09:15:00 | 2119.92 | 2000.10 | 1935.11 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-09-10 09:15:00 | 2121.24 | 2000.10 | 1935.11 | Target hit (10%) qty=1.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 1987.00 | 2038.00 | 1981.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:30:00 | 1991.10 | 2038.00 | 1981.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 1978.00 | 2037.40 | 1981.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:00:00 | 1978.00 | 2037.40 | 1981.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 1975.00 | 2036.78 | 1981.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:30:00 | 1978.50 | 2036.78 | 1981.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 12:15:00 | 1975.00 | 2036.17 | 1981.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 12:30:00 | 1976.20 | 2036.17 | 1981.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 1974.30 | 2033.11 | 1981.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:00:00 | 1974.30 | 2033.11 | 1981.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 1964.40 | 2032.42 | 1981.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:45:00 | 1960.50 | 2032.42 | 1981.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 1983.90 | 2017.68 | 1978.89 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 15:15:00 | 1855.00 | 1955.98 | 1956.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 14:15:00 | 1847.00 | 1948.52 | 1952.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 10:15:00 | 1909.20 | 1884.52 | 1911.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 10:15:00 | 1909.20 | 1884.52 | 1911.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 1909.20 | 1884.52 | 1911.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:00:00 | 1909.20 | 1884.52 | 1911.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 1913.00 | 1884.80 | 1911.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:30:00 | 1913.50 | 1884.80 | 1911.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 1906.00 | 1885.01 | 1911.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:30:00 | 1910.40 | 1885.01 | 1911.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 1910.00 | 1885.41 | 1911.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:00:00 | 1910.00 | 1885.41 | 1911.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 1901.90 | 1885.58 | 1910.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 12:15:00 | 1923.20 | 1885.58 | 1910.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 1925.00 | 1885.97 | 1911.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 12:45:00 | 1929.50 | 1885.97 | 1911.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 1910.00 | 1886.21 | 1911.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 14:45:00 | 1903.40 | 1886.28 | 1910.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 10:30:00 | 1900.60 | 1886.77 | 1910.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 1947.70 | 1879.23 | 1900.43 | SL hit (close>static) qty=1.00 sl=1925.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 1947.70 | 1879.23 | 1900.43 | SL hit (close>static) qty=1.00 sl=1925.60 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 09:15:00 | 1895.70 | 1912.66 | 1914.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 10:15:00 | 1932.50 | 1912.86 | 1914.35 | SL hit (close>static) qty=1.00 sl=1925.60 alert=retest2 |

### Cycle 3 — BUY (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 13:15:00 | 1949.40 | 1915.78 | 1915.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 10:15:00 | 1961.00 | 1917.33 | 1916.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 14:15:00 | 2005.00 | 2006.95 | 1973.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-09 15:00:00 | 2005.00 | 2006.95 | 1973.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 1981.70 | 2006.66 | 1973.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 14:45:00 | 2019.00 | 2006.40 | 1974.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 09:15:00 | 1961.90 | 2006.07 | 1974.23 | SL hit (close<static) qty=1.00 sl=1967.60 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 09:15:00 | 2027.30 | 2005.16 | 1974.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 10:15:00 | 2021.80 | 2005.23 | 1974.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 10:00:00 | 2024.80 | 2006.10 | 1976.38 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 1967.40 | 2004.93 | 1977.80 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-20 09:15:00 | 1967.40 | 2004.93 | 1977.80 | SL hit (close<static) qty=1.00 sl=1967.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-20 09:15:00 | 1967.40 | 2004.93 | 1977.80 | SL hit (close<static) qty=1.00 sl=1967.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-20 09:15:00 | 1967.40 | 2004.93 | 1977.80 | SL hit (close<static) qty=1.00 sl=1967.60 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 1969.60 | 2004.93 | 1977.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 1968.50 | 2004.57 | 1977.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 11:30:00 | 1980.70 | 2004.25 | 1977.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 13:15:00 | 1941.30 | 2003.20 | 1977.46 | SL hit (close<static) qty=1.00 sl=1956.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 09:45:00 | 1992.70 | 1997.33 | 1975.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 11:00:00 | 1978.10 | 1997.14 | 1975.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 11:45:00 | 1977.20 | 1996.95 | 1975.73 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 1980.30 | 1996.78 | 1975.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 12:30:00 | 1980.30 | 1996.78 | 1975.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 1978.20 | 1996.60 | 1975.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 13:45:00 | 1981.20 | 1996.60 | 1975.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 1992.20 | 1996.56 | 1975.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 15:15:00 | 2004.00 | 1996.56 | 1975.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-23 12:30:00 | 1993.20 | 1996.66 | 1976.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 14:15:00 | 1966.80 | 1996.15 | 1976.36 | SL hit (close<static) qty=1.00 sl=1974.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-23 14:15:00 | 1966.80 | 1996.15 | 1976.36 | SL hit (close<static) qty=1.00 sl=1974.20 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 09:30:00 | 2004.70 | 1995.87 | 1976.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 10:15:00 | 1973.10 | 1995.64 | 1976.39 | SL hit (close<static) qty=1.00 sl=1974.20 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 14:45:00 | 1996.60 | 1995.21 | 1976.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 1983.50 | 1994.87 | 1977.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 11:00:00 | 2005.90 | 1994.98 | 1977.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 12:30:00 | 2009.60 | 1995.26 | 1977.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 11:15:00 | 2006.60 | 1995.13 | 1978.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 12:00:00 | 2006.50 | 1995.24 | 1978.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 1975.00 | 1995.77 | 1978.87 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-01 09:15:00 | 1975.00 | 1995.77 | 1978.87 | SL hit (close<static) qty=1.00 sl=1975.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-01 09:15:00 | 1975.00 | 1995.77 | 1978.87 | SL hit (close<static) qty=1.00 sl=1975.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-01 09:15:00 | 1975.00 | 1995.77 | 1978.87 | SL hit (close<static) qty=1.00 sl=1975.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-01 09:15:00 | 1975.00 | 1995.77 | 1978.87 | SL hit (close<static) qty=1.00 sl=1975.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 1971.30 | 1995.29 | 1978.87 | SL hit (close<static) qty=1.00 sl=1974.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 1947.40 | 1994.43 | 1978.61 | SL hit (close<static) qty=1.00 sl=1956.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 1947.40 | 1994.43 | 1978.61 | SL hit (close<static) qty=1.00 sl=1956.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 1947.40 | 1994.43 | 1978.61 | SL hit (close<static) qty=1.00 sl=1956.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 13:30:00 | 2011.30 | 1977.71 | 1972.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 14:00:00 | 2012.60 | 1977.71 | 1972.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 09:45:00 | 2016.90 | 1978.84 | 1973.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 10:15:00 | 2018.80 | 1982.00 | 1974.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-10 12:15:00 | 2212.43 | 2050.01 | 2019.04 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-03-10 12:15:00 | 2213.86 | 2050.01 | 2019.04 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-03-10 12:15:00 | 2218.59 | 2050.01 | 2019.04 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-03-10 12:15:00 | 2220.68 | 2050.01 | 2019.04 | Target hit (10%) qty=1.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 1989.80 | 2117.93 | 2075.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 09:45:00 | 1981.80 | 2117.93 | 2075.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 2031.40 | 2117.07 | 2075.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:30:00 | 2048.90 | 2116.51 | 2075.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 14:15:00 | 2253.79 | 2134.24 | 2094.05 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-09-01 09:15:00 | 1931.20 | 2025-09-01 11:15:00 | 1905.60 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-09-01 14:30:00 | 1926.00 | 2025-09-09 13:15:00 | 2118.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-01 15:00:00 | 1927.20 | 2025-09-10 09:15:00 | 2119.92 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-02 09:30:00 | 1928.40 | 2025-09-10 09:15:00 | 2121.24 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-11-14 14:45:00 | 1903.40 | 2025-11-27 09:15:00 | 1947.70 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-11-17 10:30:00 | 1900.60 | 2025-11-27 09:15:00 | 1947.70 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-12-09 09:15:00 | 1895.70 | 2025-12-09 10:15:00 | 1932.50 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2026-01-12 14:45:00 | 2019.00 | 2026-01-13 09:15:00 | 1961.90 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2026-01-14 09:15:00 | 2027.30 | 2026-01-20 09:15:00 | 1967.40 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2026-01-14 10:15:00 | 2021.80 | 2026-01-20 09:15:00 | 1967.40 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2026-01-16 10:00:00 | 2024.80 | 2026-01-20 09:15:00 | 1967.40 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2026-01-20 11:30:00 | 1980.70 | 2026-01-20 13:15:00 | 1941.30 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2026-01-22 09:45:00 | 1992.70 | 2026-01-23 14:15:00 | 1966.80 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2026-01-22 11:00:00 | 1978.10 | 2026-01-23 14:15:00 | 1966.80 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2026-01-22 11:45:00 | 1977.20 | 2026-01-27 10:15:00 | 1973.10 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2026-01-22 15:15:00 | 2004.00 | 2026-02-01 09:15:00 | 1975.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2026-01-23 12:30:00 | 1993.20 | 2026-02-01 09:15:00 | 1975.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-01-27 09:30:00 | 2004.70 | 2026-02-01 09:15:00 | 1975.00 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2026-01-27 14:45:00 | 1996.60 | 2026-02-01 09:15:00 | 1975.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2026-01-29 11:00:00 | 2005.90 | 2026-02-01 12:15:00 | 1971.30 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2026-01-29 12:30:00 | 2009.60 | 2026-02-01 14:15:00 | 1947.40 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest2 | 2026-01-30 11:15:00 | 2006.60 | 2026-02-01 14:15:00 | 1947.40 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2026-01-30 12:00:00 | 2006.50 | 2026-02-01 14:15:00 | 1947.40 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2026-02-11 13:30:00 | 2011.30 | 2026-03-10 12:15:00 | 2212.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-11 14:00:00 | 2012.60 | 2026-03-10 12:15:00 | 2213.86 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-12 09:45:00 | 2016.90 | 2026-03-10 12:15:00 | 2218.59 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-13 10:15:00 | 2018.80 | 2026-03-10 12:15:00 | 2220.68 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 11:30:00 | 2048.90 | 2026-04-15 14:15:00 | 2253.79 | TARGET_HIT | 1.00 | 10.00% |
