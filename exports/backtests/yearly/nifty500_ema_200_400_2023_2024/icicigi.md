# ICICI Lombard General Insurance Company Ltd. (ICICIGI)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1820.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 2 |
| ALERT3 | 68 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 62 |
| PARTIAL | 12 |
| TARGET_HIT | 12 |
| STOP_HIT | 50 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 74 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 28 / 46
- **Target hits / Stop hits / Partials:** 12 / 50 / 12
- **Avg / median % per leg:** 1.35% / -1.00%
- **Sum % (uncompounded):** 100.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 4 | 22.2% | 4 | 14 | 0 | 1.20% | 21.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 18 | 4 | 22.2% | 4 | 14 | 0 | 1.20% | 21.7% |
| SELL (all) | 56 | 24 | 42.9% | 8 | 36 | 12 | 1.40% | 78.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 56 | 24 | 42.9% | 8 | 36 | 12 | 1.40% | 78.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 74 | 28 | 37.8% | 12 | 50 | 12 | 1.35% | 100.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 15:15:00 | 1543.95 | 1641.15 | 1641.52 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-06-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-10 14:15:00 | 1681.15 | 1641.96 | 1641.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-14 09:15:00 | 1711.70 | 1646.08 | 1643.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 10:15:00 | 2096.35 | 2101.65 | 2005.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 11:00:00 | 2096.35 | 2101.65 | 2005.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 2063.05 | 2149.23 | 2076.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-08 12:00:00 | 2063.05 | 2149.23 | 2076.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 2075.60 | 2148.49 | 2076.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-08 13:15:00 | 2087.35 | 2148.49 | 2076.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-08 14:15:00 | 2085.00 | 2147.80 | 2076.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-09 12:30:00 | 2103.25 | 2144.78 | 2076.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-09 15:15:00 | 2092.95 | 2143.74 | 2076.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 2079.65 | 2139.88 | 2077.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 14:45:00 | 2081.90 | 2139.88 | 2077.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 2085.00 | 2139.34 | 2077.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:15:00 | 2084.40 | 2139.34 | 2077.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 2080.00 | 2138.75 | 2077.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:45:00 | 2082.75 | 2138.75 | 2077.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 2076.95 | 2138.13 | 2077.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:00:00 | 2076.95 | 2138.13 | 2077.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 2072.00 | 2137.47 | 2077.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 13:00:00 | 2082.80 | 2136.93 | 2077.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-11 13:15:00 | 2066.65 | 2136.23 | 2077.11 | SL hit (close<static) qty=1.00 sl=2069.10 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 12:15:00 | 1935.50 | 2047.12 | 2047.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 13:15:00 | 1919.55 | 2045.85 | 2046.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 09:15:00 | 1919.90 | 1905.57 | 1952.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-04 09:30:00 | 1917.00 | 1905.57 | 1952.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 1950.70 | 1907.42 | 1952.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 15:00:00 | 1950.70 | 1907.42 | 1952.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 15:15:00 | 1948.00 | 1907.82 | 1952.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 09:15:00 | 1945.00 | 1907.82 | 1952.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 14:45:00 | 1922.75 | 1910.05 | 1952.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 11:15:00 | 1945.65 | 1911.08 | 1951.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 09:15:00 | 1940.00 | 1912.78 | 1951.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 1951.65 | 1913.17 | 1951.76 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-09 10:15:00 | 1967.00 | 1913.70 | 1951.83 | SL hit (close>static) qty=1.00 sl=1952.20 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 09:15:00 | 1862.00 | 1790.93 | 1790.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 11:15:00 | 1909.30 | 1829.47 | 1814.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 13:15:00 | 1839.40 | 1841.25 | 1823.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-26 14:00:00 | 1839.40 | 1841.25 | 1823.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1846.10 | 1841.34 | 1823.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 13:30:00 | 1858.40 | 1841.53 | 1824.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 14:00:00 | 1859.50 | 1841.53 | 1824.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 11:45:00 | 1855.40 | 1841.99 | 1825.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 09:15:00 | 1859.30 | 1842.27 | 1825.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-30 10:15:00 | 2040.94 | 1932.56 | 1891.44 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 10:15:00 | 1847.80 | 1923.02 | 1923.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-01 10:15:00 | 1832.50 | 1917.64 | 1920.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 12:15:00 | 1898.10 | 1883.97 | 1899.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 12:15:00 | 1898.10 | 1883.97 | 1899.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 1898.10 | 1883.97 | 1899.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 13:00:00 | 1898.10 | 1883.97 | 1899.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 1893.00 | 1884.18 | 1899.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:30:00 | 1898.60 | 1884.18 | 1899.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 1900.10 | 1884.34 | 1899.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 09:45:00 | 1889.20 | 1884.35 | 1899.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 11:15:00 | 1908.40 | 1885.76 | 1899.81 | SL hit (close>static) qty=1.00 sl=1907.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 15:15:00 | 2007.20 | 1900.19 | 1900.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 2027.80 | 1908.38 | 1904.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 13:15:00 | 1995.70 | 1996.81 | 1966.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-24 14:00:00 | 1995.70 | 1996.81 | 1966.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1970.70 | 1996.28 | 1969.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 12:45:00 | 1981.60 | 1993.76 | 1969.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 09:15:00 | 1956.40 | 1993.32 | 1969.96 | SL hit (close<static) qty=1.00 sl=1965.00 alert=retest2 |

### Cycle 7 — SELL (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 13:15:00 | 1950.00 | 1960.04 | 1960.07 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 1979.90 | 1960.23 | 1960.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 11:15:00 | 1988.40 | 1960.51 | 1960.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 1946.50 | 1965.51 | 1962.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 1946.50 | 1965.51 | 1962.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1946.50 | 1965.51 | 1962.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 1946.50 | 1965.51 | 1962.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 1921.30 | 1965.07 | 1962.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 1921.30 | 1965.07 | 1962.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 11:15:00 | 1895.40 | 1960.57 | 1960.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 12:15:00 | 1874.00 | 1951.38 | 1955.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 13:15:00 | 1875.50 | 1873.77 | 1904.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-05 13:30:00 | 1876.20 | 1873.77 | 1904.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 1899.90 | 1873.24 | 1901.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:00:00 | 1899.90 | 1873.24 | 1901.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 1899.00 | 1873.50 | 1901.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:30:00 | 1902.50 | 1873.50 | 1901.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 1905.90 | 1873.82 | 1901.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 13:00:00 | 1905.90 | 1873.82 | 1901.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 1911.30 | 1874.20 | 1901.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 13:45:00 | 1910.00 | 1874.20 | 1901.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1895.40 | 1875.04 | 1901.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 15:15:00 | 1892.80 | 1907.44 | 1911.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 13:45:00 | 1890.70 | 1906.96 | 1911.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 14:15:00 | 1890.90 | 1906.96 | 1911.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 14:45:00 | 1892.70 | 1906.83 | 1911.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 15:15:00 | 1881.30 | 1896.84 | 1905.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 09:15:00 | 1876.50 | 1896.60 | 1905.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 15:00:00 | 1875.00 | 1895.84 | 1904.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 10:15:00 | 1798.16 | 1880.83 | 1895.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 10:15:00 | 1796.36 | 1880.83 | 1895.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 10:15:00 | 1798.07 | 1880.83 | 1895.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 09:15:00 | 1796.16 | 1877.45 | 1892.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 14:15:00 | 1782.67 | 1868.54 | 1887.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 14:15:00 | 1781.25 | 1868.54 | 1887.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-24 10:15:00 | 1703.52 | 1855.13 | 1879.71 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-10-08 13:15:00 | 2087.35 | 2024-10-11 13:15:00 | 2066.65 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-10-08 14:15:00 | 2085.00 | 2024-10-17 10:15:00 | 2052.60 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-10-09 12:30:00 | 2103.25 | 2024-10-17 10:15:00 | 2052.60 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2024-10-09 15:15:00 | 2092.95 | 2024-10-17 10:15:00 | 2052.60 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-10-11 13:00:00 | 2082.80 | 2024-10-17 10:15:00 | 2052.60 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-10-14 11:30:00 | 2086.95 | 2024-10-17 10:15:00 | 2052.60 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2024-12-05 09:15:00 | 1945.00 | 2024-12-09 10:15:00 | 1967.00 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-12-05 14:45:00 | 1922.75 | 2024-12-09 10:15:00 | 1967.00 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2024-12-06 11:15:00 | 1945.65 | 2024-12-09 10:15:00 | 1967.00 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-12-09 09:15:00 | 1940.00 | 2024-12-09 10:15:00 | 1967.00 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-12-10 12:30:00 | 1934.35 | 2024-12-12 15:15:00 | 1968.00 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-12-10 13:30:00 | 1935.00 | 2024-12-12 15:15:00 | 1968.00 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-12-10 14:15:00 | 1935.20 | 2024-12-12 15:15:00 | 1968.00 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-12-11 12:30:00 | 1931.75 | 2024-12-12 15:15:00 | 1968.00 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-01-16 11:15:00 | 1888.80 | 2025-01-16 15:15:00 | 1914.70 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-01-16 11:45:00 | 1884.05 | 2025-01-16 15:15:00 | 1914.70 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-01-22 09:45:00 | 1894.35 | 2025-01-27 09:15:00 | 1799.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-22 09:45:00 | 1894.35 | 2025-01-30 12:15:00 | 1863.35 | STOP_HIT | 0.50 | 1.64% |
| SELL | retest2 | 2025-02-01 11:45:00 | 1878.40 | 2025-02-11 11:15:00 | 1784.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 10:00:00 | 1796.00 | 2025-02-24 11:15:00 | 1706.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-01 11:45:00 | 1878.40 | 2025-02-25 11:15:00 | 1690.56 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-11 10:00:00 | 1796.00 | 2025-03-04 11:15:00 | 1616.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-25 10:00:00 | 1790.00 | 2025-04-11 15:15:00 | 1703.87 | PARTIAL | 0.50 | 4.81% |
| SELL | retest2 | 2025-03-25 10:00:00 | 1790.00 | 2025-04-15 10:15:00 | 1773.00 | STOP_HIT | 0.50 | 0.95% |
| SELL | retest2 | 2025-03-25 14:15:00 | 1791.10 | 2025-04-15 12:15:00 | 1809.70 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-03-28 09:30:00 | 1793.55 | 2025-04-15 12:15:00 | 1809.70 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-04-11 09:15:00 | 1759.00 | 2025-04-15 12:15:00 | 1809.70 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-04-11 10:15:00 | 1754.45 | 2025-04-16 09:15:00 | 1807.00 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-04-15 10:00:00 | 1759.30 | 2025-04-16 14:15:00 | 1810.90 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2025-04-16 09:15:00 | 1759.60 | 2025-04-21 10:15:00 | 1832.30 | STOP_HIT | 1.00 | -4.13% |
| SELL | retest2 | 2025-04-16 11:45:00 | 1790.00 | 2025-04-21 10:15:00 | 1832.30 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-04-17 09:45:00 | 1790.00 | 2025-04-21 10:15:00 | 1832.30 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-04-17 10:45:00 | 1785.50 | 2025-04-25 09:15:00 | 1862.00 | STOP_HIT | 1.00 | -4.28% |
| SELL | retest2 | 2025-04-17 13:15:00 | 1790.00 | 2025-04-25 09:15:00 | 1862.00 | STOP_HIT | 1.00 | -4.02% |
| BUY | retest2 | 2025-05-28 13:30:00 | 1858.40 | 2025-06-30 10:15:00 | 2040.94 | TARGET_HIT | 1.00 | 9.82% |
| BUY | retest2 | 2025-05-28 14:00:00 | 1859.50 | 2025-06-30 11:15:00 | 2044.24 | TARGET_HIT | 1.00 | 9.93% |
| BUY | retest2 | 2025-05-29 11:45:00 | 1855.40 | 2025-06-30 11:15:00 | 2045.45 | TARGET_HIT | 1.00 | 10.24% |
| BUY | retest2 | 2025-05-30 09:15:00 | 1859.30 | 2025-06-30 11:15:00 | 2045.23 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-18 09:15:00 | 1966.10 | 2025-08-19 09:15:00 | 1928.20 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-08-19 12:00:00 | 1950.10 | 2025-08-22 11:15:00 | 1937.10 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-08-19 14:00:00 | 1946.20 | 2025-08-22 11:15:00 | 1937.10 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-08-19 14:30:00 | 1949.90 | 2025-08-22 11:15:00 | 1937.10 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-09-15 09:45:00 | 1889.20 | 2025-09-16 11:15:00 | 1908.40 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-09-17 12:45:00 | 1885.60 | 2025-09-26 09:15:00 | 1900.10 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-09-23 11:15:00 | 1889.20 | 2025-10-01 11:15:00 | 1900.00 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-09-23 12:45:00 | 1889.10 | 2025-10-03 09:15:00 | 1908.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-09-25 09:15:00 | 1861.00 | 2025-10-03 09:15:00 | 1908.00 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2025-10-01 09:45:00 | 1866.20 | 2025-10-03 09:15:00 | 1908.00 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-10-09 09:15:00 | 1860.30 | 2025-10-15 09:15:00 | 2000.00 | STOP_HIT | 1.00 | -7.51% |
| SELL | retest2 | 2025-10-10 13:00:00 | 1870.00 | 2025-10-15 09:15:00 | 2000.00 | STOP_HIT | 1.00 | -6.95% |
| BUY | retest2 | 2025-12-01 12:45:00 | 1981.60 | 2025-12-02 09:15:00 | 1956.40 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-12-02 12:30:00 | 1982.00 | 2025-12-03 09:15:00 | 1955.50 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-12-04 10:30:00 | 1981.40 | 2025-12-08 09:15:00 | 1961.50 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-12-04 13:00:00 | 1981.50 | 2025-12-08 09:15:00 | 1961.50 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2026-02-27 15:15:00 | 1892.80 | 2026-03-18 10:15:00 | 1798.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 13:45:00 | 1890.70 | 2026-03-18 10:15:00 | 1796.36 | PARTIAL | 0.50 | 4.99% |
| SELL | retest2 | 2026-03-02 14:15:00 | 1890.90 | 2026-03-18 10:15:00 | 1798.07 | PARTIAL | 0.50 | 4.91% |
| SELL | retest2 | 2026-03-02 14:45:00 | 1892.70 | 2026-03-19 09:15:00 | 1796.16 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2026-03-11 09:15:00 | 1876.50 | 2026-03-20 14:15:00 | 1782.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 15:00:00 | 1875.00 | 2026-03-20 14:15:00 | 1781.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 15:15:00 | 1892.80 | 2026-03-24 10:15:00 | 1703.52 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-02 13:45:00 | 1890.70 | 2026-03-24 10:15:00 | 1703.43 | TARGET_HIT | 0.50 | 9.90% |
| SELL | retest2 | 2026-03-02 14:15:00 | 1890.90 | 2026-03-30 09:15:00 | 1701.63 | TARGET_HIT | 0.50 | 10.01% |
| SELL | retest2 | 2026-03-02 14:45:00 | 1892.70 | 2026-03-30 09:15:00 | 1701.81 | TARGET_HIT | 0.50 | 10.09% |
| SELL | retest2 | 2026-03-11 09:15:00 | 1876.50 | 2026-03-30 09:15:00 | 1688.85 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-11 15:00:00 | 1875.00 | 2026-03-30 09:15:00 | 1687.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-16 10:30:00 | 1878.50 | 2026-04-24 11:15:00 | 1784.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-17 09:15:00 | 1875.00 | 2026-04-24 11:15:00 | 1781.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-16 10:30:00 | 1878.50 | 2026-05-06 10:15:00 | 1797.50 | STOP_HIT | 0.50 | 4.31% |
| SELL | retest2 | 2026-04-17 09:15:00 | 1875.00 | 2026-05-06 10:15:00 | 1797.50 | STOP_HIT | 0.50 | 4.13% |
| SELL | retest2 | 2026-04-22 10:15:00 | 1824.00 | 2026-05-07 11:15:00 | 1846.70 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-04-22 12:00:00 | 1822.90 | 2026-05-07 11:15:00 | 1846.70 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-04-23 09:15:00 | 1811.70 | 2026-05-07 11:15:00 | 1846.70 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2026-04-23 13:30:00 | 1818.50 | 2026-05-07 11:15:00 | 1846.70 | STOP_HIT | 1.00 | -1.55% |
