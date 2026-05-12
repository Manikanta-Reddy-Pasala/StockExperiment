# BEML Ltd. (BEML)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1952.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT2_SKIP | 3 |
| ALERT3 | 35 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 33 |
| PARTIAL | 10 |
| TARGET_HIT | 6 |
| STOP_HIT | 27 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 43 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 21
- **Target hits / Stop hits / Partials:** 6 / 27 / 10
- **Avg / median % per leg:** 1.49% / 1.17%
- **Sum % (uncompounded):** 63.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 1 | 14.3% | 1 | 6 | 0 | -1.04% | -7.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 1 | 14.3% | 1 | 6 | 0 | -1.04% | -7.3% |
| SELL (all) | 36 | 21 | 58.3% | 5 | 21 | 10 | 1.98% | 71.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 36 | 21 | 58.3% | 5 | 21 | 10 | 1.98% | 71.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 43 | 22 | 51.2% | 6 | 27 | 10 | 1.49% | 64.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 09:15:00 | 1911.00 | 2132.85 | 2133.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 10:15:00 | 1880.43 | 2130.33 | 2132.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 09:15:00 | 2032.58 | 2024.25 | 2068.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-04 10:00:00 | 2032.58 | 2024.25 | 2068.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 10:15:00 | 2037.38 | 2024.38 | 2068.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 10:30:00 | 2049.45 | 2024.38 | 2068.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 14:15:00 | 2049.32 | 2024.68 | 2068.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 14:45:00 | 2053.00 | 2024.68 | 2068.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 2044.00 | 2025.16 | 2067.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 14:30:00 | 2028.35 | 2025.82 | 2067.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 15:15:00 | 2025.00 | 2025.82 | 2067.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 1926.93 | 2022.02 | 2063.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 1923.75 | 2022.02 | 2063.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-09-19 09:15:00 | 1825.51 | 1988.89 | 2034.65 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 09:15:00 | 2064.30 | 1969.80 | 1969.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-11 10:15:00 | 2109.23 | 1971.19 | 1970.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-12 14:15:00 | 1968.65 | 1977.81 | 1973.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 14:15:00 | 1968.65 | 1977.81 | 1973.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 14:15:00 | 1968.65 | 1977.81 | 1973.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 15:00:00 | 1968.65 | 1977.81 | 1973.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 15:15:00 | 1976.03 | 1977.80 | 1973.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-13 09:15:00 | 1990.00 | 1977.80 | 1973.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-13 09:15:00 | 1951.53 | 1977.54 | 1973.64 | SL hit (close<static) qty=1.00 sl=1956.30 alert=retest2 |

### Cycle 3 — SELL (started 2024-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 14:15:00 | 1894.00 | 1969.99 | 1969.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 09:15:00 | 1869.50 | 1968.22 | 1969.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 1995.98 | 1947.18 | 1957.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 09:15:00 | 1995.98 | 1947.18 | 1957.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 1995.98 | 1947.18 | 1957.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:30:00 | 2020.00 | 1947.18 | 1957.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 10:15:00 | 1991.38 | 1947.62 | 1957.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 10:30:00 | 2000.43 | 1947.62 | 1957.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2024-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 11:15:00 | 2145.00 | 1967.76 | 1967.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 09:15:00 | 2174.93 | 2022.14 | 1997.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 12:15:00 | 2086.35 | 2109.62 | 2056.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-20 13:00:00 | 2086.35 | 2109.62 | 2056.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 13:15:00 | 2062.07 | 2109.15 | 2056.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 14:00:00 | 2062.07 | 2109.15 | 2056.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 2035.00 | 2108.41 | 2056.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 2035.00 | 2108.41 | 2056.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 2040.90 | 2107.74 | 2056.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:15:00 | 2028.50 | 2107.74 | 2056.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 2041.98 | 2106.19 | 2056.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 11:15:00 | 2065.13 | 2106.19 | 2056.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 09:45:00 | 2060.80 | 2086.07 | 2052.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-31 09:15:00 | 1990.40 | 2082.15 | 2051.22 | SL hit (close<static) qty=1.00 sl=1997.23 alert=retest2 |

### Cycle 5 — SELL (started 2025-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 11:15:00 | 1869.50 | 2031.14 | 2031.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 12:15:00 | 1855.00 | 2029.38 | 2030.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 10:15:00 | 1885.08 | 1874.10 | 1933.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-31 11:00:00 | 1885.08 | 1874.10 | 1933.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 14:15:00 | 1924.03 | 1875.45 | 1933.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 14:45:00 | 1908.60 | 1875.45 | 1933.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 15:15:00 | 1933.98 | 1876.03 | 1933.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 09:15:00 | 1976.55 | 1876.03 | 1933.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 1965.13 | 1876.92 | 1933.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 11:00:00 | 1940.45 | 1877.55 | 1933.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 11:45:00 | 1919.60 | 1878.11 | 1933.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 12:15:00 | 1843.43 | 1877.84 | 1933.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 12:15:00 | 1823.62 | 1877.84 | 1933.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-03 09:15:00 | 1746.40 | 1875.69 | 1931.04 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 1732.50 | 1547.48 | 1547.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 1790.00 | 1566.41 | 1556.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 13:15:00 | 2206.85 | 2211.87 | 2077.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 14:00:00 | 2206.85 | 2211.87 | 2077.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 2093.15 | 2198.96 | 2090.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:00:00 | 2093.15 | 2198.96 | 2090.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 2089.55 | 2197.87 | 2090.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:30:00 | 2090.15 | 2197.87 | 2090.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 2096.50 | 2196.86 | 2090.33 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 13:15:00 | 1922.15 | 2040.72 | 2040.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 1913.30 | 2037.04 | 2038.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 12:15:00 | 2050.90 | 2023.70 | 2031.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 12:15:00 | 2050.90 | 2023.70 | 2031.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 12:15:00 | 2050.90 | 2023.70 | 2031.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 13:00:00 | 2050.90 | 2023.70 | 2031.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 2030.50 | 2023.76 | 2031.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 09:15:00 | 2023.40 | 2025.63 | 2032.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 10:00:00 | 2016.05 | 2025.53 | 2032.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 10:15:00 | 2062.95 | 2024.09 | 2031.28 | SL hit (close>static) qty=1.00 sl=2054.45 alert=retest2 |

### Cycle 8 — BUY (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 15:15:00 | 2187.50 | 2037.21 | 2036.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 10:15:00 | 2245.00 | 2084.24 | 2062.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 2096.35 | 2106.21 | 2077.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-26 14:00:00 | 2096.35 | 2106.21 | 2077.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 2081.20 | 2106.08 | 2078.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:00:00 | 2081.20 | 2106.08 | 2078.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 2051.45 | 2105.53 | 2077.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 12:00:00 | 2051.45 | 2105.53 | 2077.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 2049.60 | 2104.98 | 2077.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 12:45:00 | 2043.95 | 2104.98 | 2077.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 2061.60 | 2101.39 | 2076.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 14:30:00 | 2068.50 | 2100.75 | 2076.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-24 11:15:00 | 2275.35 | 2158.81 | 2121.90 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 2031.10 | 2110.15 | 2110.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 2011.00 | 2109.17 | 2109.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 09:15:00 | 1857.30 | 1809.05 | 1905.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:45:00 | 1868.70 | 1809.05 | 1905.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1894.50 | 1815.18 | 1902.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 15:00:00 | 1853.40 | 1823.60 | 1901.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 09:30:00 | 1867.30 | 1824.92 | 1898.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 14:45:00 | 1859.90 | 1830.78 | 1895.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 10:15:00 | 1860.70 | 1835.65 | 1894.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1760.73 | 1835.25 | 1887.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1773.93 | 1835.25 | 1887.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1766.90 | 1835.25 | 1887.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1767.66 | 1835.25 | 1887.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-14 11:15:00 | 1831.80 | 1827.23 | 1878.92 | SL hit (close>ema200) qty=0.50 sl=1827.23 alert=retest2 |

### Cycle 10 — BUY (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 13:15:00 | 1809.50 | 1675.32 | 1675.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 14:15:00 | 1842.50 | 1676.98 | 1675.91 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-09-05 14:30:00 | 2028.35 | 2024-09-09 09:15:00 | 1926.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-05 15:15:00 | 2025.00 | 2024-09-09 09:15:00 | 1923.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-05 14:30:00 | 2028.35 | 2024-09-19 09:15:00 | 1825.51 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-05 15:15:00 | 2025.00 | 2024-09-19 09:15:00 | 1822.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-18 13:00:00 | 2028.00 | 2024-10-22 12:15:00 | 1926.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 13:00:00 | 2028.00 | 2024-10-22 12:15:00 | 1928.75 | STOP_HIT | 0.50 | 4.89% |
| SELL | retest2 | 2024-10-18 14:00:00 | 2023.20 | 2024-10-22 13:15:00 | 1922.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 14:00:00 | 2023.20 | 2024-10-22 13:15:00 | 1932.98 | STOP_HIT | 0.50 | 4.46% |
| SELL | retest2 | 2024-10-22 14:15:00 | 1921.00 | 2024-10-30 10:15:00 | 1957.98 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2024-10-23 14:30:00 | 1926.58 | 2024-10-30 10:15:00 | 1957.98 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-10-24 09:30:00 | 1927.38 | 2024-10-30 10:15:00 | 1957.98 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-10-25 09:15:00 | 1908.00 | 2024-10-30 10:15:00 | 1957.98 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2024-10-25 10:30:00 | 1860.20 | 2024-10-30 10:15:00 | 1957.98 | STOP_HIT | 1.00 | -5.26% |
| SELL | retest2 | 2024-10-25 11:30:00 | 1864.73 | 2024-10-30 10:15:00 | 1957.98 | STOP_HIT | 1.00 | -5.00% |
| SELL | retest2 | 2024-10-25 12:15:00 | 1859.38 | 2024-10-30 10:15:00 | 1957.98 | STOP_HIT | 1.00 | -5.30% |
| SELL | retest2 | 2024-10-25 13:45:00 | 1860.78 | 2024-10-30 10:15:00 | 1957.98 | STOP_HIT | 1.00 | -5.22% |
| BUY | retest2 | 2024-11-13 09:15:00 | 1990.00 | 2024-11-13 09:15:00 | 1951.53 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-12-23 11:15:00 | 2065.13 | 2024-12-31 09:15:00 | 1990.40 | STOP_HIT | 1.00 | -3.62% |
| BUY | retest2 | 2024-12-30 09:45:00 | 2060.80 | 2024-12-31 09:15:00 | 1990.40 | STOP_HIT | 1.00 | -3.42% |
| BUY | retest2 | 2025-01-01 09:15:00 | 2054.23 | 2025-01-06 09:15:00 | 2033.38 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-01-01 11:00:00 | 2055.90 | 2025-01-06 12:15:00 | 1993.63 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2025-01-02 09:15:00 | 2082.32 | 2025-01-06 12:15:00 | 1993.63 | STOP_HIT | 1.00 | -4.26% |
| SELL | retest2 | 2025-02-01 11:00:00 | 1940.45 | 2025-02-01 12:15:00 | 1843.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-01 11:45:00 | 1919.60 | 2025-02-01 12:15:00 | 1823.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-01 11:00:00 | 1940.45 | 2025-02-03 09:15:00 | 1746.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-01 11:45:00 | 1919.60 | 2025-02-03 09:15:00 | 1727.64 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-04 09:15:00 | 2023.40 | 2025-09-05 10:15:00 | 2062.95 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-09-04 10:00:00 | 2016.05 | 2025-09-05 10:15:00 | 2062.95 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-09-09 14:30:00 | 2024.00 | 2025-09-12 09:15:00 | 2086.70 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2025-09-10 14:45:00 | 2024.50 | 2025-09-12 09:15:00 | 2086.70 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2025-09-11 11:30:00 | 2030.60 | 2025-09-12 09:15:00 | 2086.70 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2025-09-30 14:30:00 | 2068.50 | 2025-10-24 11:15:00 | 2275.35 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-12-29 15:00:00 | 1853.40 | 2026-01-12 09:15:00 | 1760.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-31 09:30:00 | 1867.30 | 2026-01-12 09:15:00 | 1773.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-02 14:45:00 | 1859.90 | 2026-01-12 09:15:00 | 1766.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 10:15:00 | 1860.70 | 2026-01-12 09:15:00 | 1767.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-29 15:00:00 | 1853.40 | 2026-01-14 11:15:00 | 1831.80 | STOP_HIT | 0.50 | 1.17% |
| SELL | retest2 | 2025-12-31 09:30:00 | 1867.30 | 2026-01-14 11:15:00 | 1831.80 | STOP_HIT | 0.50 | 1.90% |
| SELL | retest2 | 2026-01-02 14:45:00 | 1859.90 | 2026-01-14 11:15:00 | 1831.80 | STOP_HIT | 0.50 | 1.51% |
| SELL | retest2 | 2026-01-06 10:15:00 | 1860.70 | 2026-01-14 11:15:00 | 1831.80 | STOP_HIT | 0.50 | 1.55% |
| SELL | retest2 | 2026-02-01 11:00:00 | 1788.30 | 2026-02-01 12:15:00 | 1609.47 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-17 09:45:00 | 1785.50 | 2026-04-28 13:15:00 | 1809.50 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-04-24 11:00:00 | 1789.50 | 2026-04-28 13:15:00 | 1809.50 | STOP_HIT | 1.00 | -1.12% |
