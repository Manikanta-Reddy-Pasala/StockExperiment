# United Breweries Ltd. (UBL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1419.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 77 |
| ALERT1 | 44 |
| ALERT2 | 44 |
| ALERT2_SKIP | 32 |
| ALERT3 | 110 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 59 |
| PARTIAL | 12 |
| TARGET_HIT | 0 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 71 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 41 / 30
- **Target hits / Stop hits / Partials:** 0 / 59 / 12
- **Avg / median % per leg:** 1.15% / 0.43%
- **Sum % (uncompounded):** 81.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 3 | 18.8% | 0 | 16 | 0 | -0.68% | -10.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 3 | 18.8% | 0 | 16 | 0 | -0.68% | -10.9% |
| SELL (all) | 55 | 38 | 69.1% | 0 | 43 | 12 | 1.68% | 92.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 55 | 38 | 69.1% | 0 | 43 | 12 | 1.68% | 92.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 71 | 41 | 57.7% | 0 | 59 | 12 | 1.15% | 81.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-12 10:15:00 | 2138.40 | 2157.52 | 2158.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-12 12:15:00 | 2126.40 | 2147.71 | 2153.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-14 09:15:00 | 2074.90 | 2074.59 | 2102.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 09:15:00 | 2074.50 | 2073.13 | 2087.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 2074.50 | 2073.13 | 2087.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-15 11:00:00 | 2058.00 | 2070.10 | 2085.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-15 15:15:00 | 2054.00 | 2075.23 | 2083.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-16 09:45:00 | 2060.50 | 2067.81 | 2078.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-19 11:45:00 | 2060.00 | 2050.67 | 2060.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 12:15:00 | 2053.00 | 2051.13 | 2060.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-19 14:15:00 | 2049.10 | 2050.93 | 2059.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-19 15:15:00 | 2045.00 | 2051.14 | 2058.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-20 10:15:00 | 2047.30 | 2050.19 | 2056.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-20 11:15:00 | 2048.40 | 2050.31 | 2056.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 2052.50 | 2039.57 | 2047.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 12:45:00 | 2027.50 | 2037.25 | 2044.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 2013.70 | 2033.83 | 2040.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 10:15:00 | 2039.50 | 2022.62 | 2021.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 2039.50 | 2022.62 | 2021.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 11:15:00 | 2045.70 | 2027.24 | 2023.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 2021.10 | 2034.16 | 2029.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 2021.10 | 2034.16 | 2029.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 2021.10 | 2034.16 | 2029.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 2021.10 | 2034.16 | 2029.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 2017.70 | 2030.87 | 2028.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:45:00 | 2017.40 | 2030.87 | 2028.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 12:15:00 | 2011.00 | 2024.52 | 2025.75 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 2043.10 | 2029.22 | 2027.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 11:15:00 | 2051.00 | 2033.97 | 2029.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 09:15:00 | 2026.80 | 2043.20 | 2037.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 2026.80 | 2043.20 | 2037.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 2026.80 | 2043.20 | 2037.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:45:00 | 2030.00 | 2043.20 | 2037.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 2022.80 | 2039.12 | 2036.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:30:00 | 2026.30 | 2039.12 | 2036.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 12:15:00 | 2013.10 | 2031.02 | 2032.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 14:15:00 | 2002.40 | 2021.99 | 2028.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 2007.00 | 1994.00 | 2006.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 2007.00 | 1994.00 | 2006.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 2007.00 | 1994.00 | 2006.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:00:00 | 2007.00 | 1994.00 | 2006.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 2001.90 | 1995.58 | 2005.65 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 10:15:00 | 2025.00 | 2011.63 | 2009.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 09:15:00 | 2036.20 | 2020.29 | 2014.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 13:15:00 | 2030.00 | 2030.52 | 2025.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-05 14:00:00 | 2030.00 | 2030.52 | 2025.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 2016.00 | 2027.10 | 2024.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 10:30:00 | 2032.30 | 2030.62 | 2026.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 15:15:00 | 2055.10 | 2062.89 | 2063.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 2055.10 | 2062.89 | 2063.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 2037.70 | 2057.74 | 2060.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 2053.80 | 2043.01 | 2050.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 14:15:00 | 2053.80 | 2043.01 | 2050.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 2053.80 | 2043.01 | 2050.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 2053.80 | 2043.01 | 2050.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 2040.40 | 2042.49 | 2049.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:30:00 | 2034.90 | 2041.15 | 2048.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 1933.15 | 1965.95 | 1991.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-19 15:15:00 | 1963.00 | 1960.15 | 1981.72 | SL hit (close>ema200) qty=0.50 sl=1960.15 alert=retest2 |

### Cycle 8 — BUY (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 11:15:00 | 1955.10 | 1940.42 | 1939.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 12:15:00 | 1963.40 | 1945.02 | 1941.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 1954.40 | 1963.11 | 1955.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 14:15:00 | 1954.40 | 1963.11 | 1955.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 1954.40 | 1963.11 | 1955.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 1954.40 | 1963.11 | 1955.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 1960.00 | 1962.49 | 1956.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 1976.00 | 1962.49 | 1956.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 12:15:00 | 1949.90 | 1965.57 | 1960.32 | SL hit (close<static) qty=1.00 sl=1950.10 alert=retest2 |

### Cycle 9 — SELL (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 12:15:00 | 1947.20 | 1958.16 | 1958.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 11:15:00 | 1935.40 | 1948.55 | 1953.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 1960.00 | 1947.28 | 1950.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 1960.00 | 1947.28 | 1950.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1960.00 | 1947.28 | 1950.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 1960.00 | 1947.28 | 1950.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 1977.10 | 1953.24 | 1952.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 12:15:00 | 1981.20 | 1963.12 | 1957.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 09:15:00 | 1971.20 | 1972.55 | 1964.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 10:00:00 | 1971.20 | 1972.55 | 1964.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 1965.00 | 1970.15 | 1964.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:00:00 | 1965.00 | 1970.15 | 1964.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 1969.30 | 1969.98 | 1965.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:45:00 | 1963.30 | 1969.98 | 1965.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1979.20 | 1985.70 | 1979.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:30:00 | 1981.90 | 1985.70 | 1979.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 1980.00 | 1984.56 | 1979.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:15:00 | 1979.30 | 1984.56 | 1979.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 1978.60 | 1983.37 | 1979.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:00:00 | 1978.60 | 1983.37 | 1979.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 1975.00 | 1981.70 | 1978.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:00:00 | 1975.00 | 1981.70 | 1978.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 1975.10 | 1980.38 | 1978.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:45:00 | 1976.00 | 1980.38 | 1978.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 1990.00 | 1982.32 | 1979.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 1972.70 | 1982.32 | 1979.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1964.60 | 1978.78 | 1978.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:00:00 | 1964.60 | 1978.78 | 1978.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 10:15:00 | 1966.90 | 1976.40 | 1977.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 12:15:00 | 1949.90 | 1969.12 | 1973.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 14:15:00 | 1970.10 | 1968.49 | 1972.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 14:15:00 | 1970.10 | 1968.49 | 1972.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 1970.10 | 1968.49 | 1972.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:45:00 | 1972.90 | 1968.49 | 1972.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1942.50 | 1963.62 | 1969.71 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 1985.20 | 1961.10 | 1960.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 10:15:00 | 2000.70 | 1979.53 | 1971.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 11:15:00 | 2009.30 | 2016.63 | 2006.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 12:00:00 | 2009.30 | 2016.63 | 2006.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 1994.30 | 2013.25 | 2006.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:00:00 | 1994.30 | 2013.25 | 2006.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 2002.50 | 2011.10 | 2006.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 15:15:00 | 2009.50 | 2011.10 | 2006.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 09:45:00 | 2010.60 | 2010.50 | 2006.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 12:45:00 | 2019.00 | 2010.43 | 2007.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 14:15:00 | 2017.90 | 2024.20 | 2024.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2025-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 14:15:00 | 2017.90 | 2024.20 | 2024.92 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2025-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-25 14:15:00 | 2034.40 | 2025.84 | 2024.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-25 15:15:00 | 2037.90 | 2028.25 | 2026.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 09:15:00 | 2021.70 | 2026.94 | 2025.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 2021.70 | 2026.94 | 2025.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 2021.70 | 2026.94 | 2025.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:00:00 | 2021.70 | 2026.94 | 2025.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — SELL (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 10:15:00 | 2014.10 | 2024.37 | 2024.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 11:15:00 | 1997.80 | 2019.06 | 2022.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 11:15:00 | 1959.00 | 1956.83 | 1970.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-31 12:00:00 | 1959.00 | 1956.83 | 1970.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 1939.00 | 1933.50 | 1942.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:30:00 | 1940.20 | 1933.50 | 1942.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1938.60 | 1934.52 | 1942.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:45:00 | 1940.70 | 1934.52 | 1942.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 1935.00 | 1934.61 | 1941.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 1932.00 | 1934.61 | 1941.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1923.40 | 1932.37 | 1939.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 11:45:00 | 1919.20 | 1925.46 | 1931.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 14:45:00 | 1920.10 | 1925.42 | 1929.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 09:45:00 | 1919.00 | 1925.75 | 1929.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 1935.30 | 1930.48 | 1930.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 1935.30 | 1930.48 | 1930.42 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 09:15:00 | 1920.40 | 1929.32 | 1929.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 09:15:00 | 1910.00 | 1920.14 | 1924.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 14:15:00 | 1920.00 | 1915.08 | 1919.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 14:15:00 | 1920.00 | 1915.08 | 1919.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 1920.00 | 1915.08 | 1919.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 15:00:00 | 1920.00 | 1915.08 | 1919.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 1920.00 | 1916.07 | 1919.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:15:00 | 1912.50 | 1916.07 | 1919.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 1921.00 | 1917.05 | 1919.79 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 12:15:00 | 1925.20 | 1922.17 | 1921.80 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 1914.30 | 1920.45 | 1921.11 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2025-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 15:15:00 | 1924.00 | 1921.37 | 1921.26 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 1914.30 | 1919.95 | 1920.62 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 13:15:00 | 1927.60 | 1922.15 | 1921.51 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 11:15:00 | 1893.70 | 1917.35 | 1919.74 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2025-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 15:15:00 | 1926.50 | 1916.67 | 1915.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 14:15:00 | 1929.10 | 1918.23 | 1916.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 09:15:00 | 1915.00 | 1918.04 | 1916.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 1915.00 | 1918.04 | 1916.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1915.00 | 1918.04 | 1916.83 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 10:15:00 | 1904.10 | 1915.26 | 1915.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 11:15:00 | 1890.00 | 1910.20 | 1913.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 11:15:00 | 1857.70 | 1855.62 | 1867.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-26 12:00:00 | 1857.70 | 1855.62 | 1867.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 1834.60 | 1850.14 | 1860.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 09:30:00 | 1822.00 | 1845.95 | 1851.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 09:45:00 | 1821.10 | 1812.52 | 1827.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 09:15:00 | 1847.60 | 1833.07 | 1832.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 1847.60 | 1833.07 | 1832.29 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2025-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 14:15:00 | 1829.90 | 1832.32 | 1832.37 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 1847.40 | 1834.70 | 1833.40 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2025-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 13:15:00 | 1828.90 | 1833.72 | 1834.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 14:15:00 | 1825.30 | 1832.03 | 1833.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1831.80 | 1831.02 | 1832.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 1831.80 | 1831.02 | 1832.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1831.80 | 1831.02 | 1832.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 13:00:00 | 1814.20 | 1824.37 | 1828.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 10:00:00 | 1809.20 | 1813.84 | 1821.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 10:00:00 | 1810.30 | 1809.12 | 1815.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 09:15:00 | 1832.90 | 1806.18 | 1803.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 1832.90 | 1806.18 | 1803.21 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 14:15:00 | 1799.60 | 1810.10 | 1811.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 12:15:00 | 1795.20 | 1803.64 | 1807.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 11:15:00 | 1800.50 | 1785.93 | 1790.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 11:15:00 | 1800.50 | 1785.93 | 1790.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 1800.50 | 1785.93 | 1790.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:45:00 | 1800.90 | 1785.93 | 1790.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 1797.20 | 1788.18 | 1791.13 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2025-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 09:15:00 | 1807.60 | 1795.15 | 1793.77 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 11:15:00 | 1786.50 | 1792.59 | 1792.80 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 14:15:00 | 1797.30 | 1793.65 | 1793.19 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 09:15:00 | 1775.80 | 1790.45 | 1791.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 10:15:00 | 1763.00 | 1784.96 | 1789.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 14:15:00 | 1783.50 | 1778.60 | 1784.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 14:15:00 | 1783.50 | 1778.60 | 1784.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 1783.50 | 1778.60 | 1784.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:00:00 | 1783.50 | 1778.60 | 1784.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 1783.70 | 1779.62 | 1784.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:15:00 | 1765.30 | 1779.62 | 1784.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 1770.50 | 1777.80 | 1782.85 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2025-09-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 12:15:00 | 1795.00 | 1783.30 | 1782.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 13:15:00 | 1798.00 | 1786.24 | 1783.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 09:15:00 | 1794.40 | 1797.68 | 1793.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 09:15:00 | 1794.40 | 1797.68 | 1793.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1794.40 | 1797.68 | 1793.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 09:15:00 | 1815.80 | 1796.87 | 1794.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 10:15:00 | 1788.80 | 1797.27 | 1797.02 | SL hit (close<static) qty=1.00 sl=1790.70 alert=retest2 |

### Cycle 37 — SELL (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 11:15:00 | 1789.10 | 1795.64 | 1796.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 1777.90 | 1787.19 | 1790.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 1760.00 | 1756.37 | 1766.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 15:00:00 | 1760.00 | 1756.37 | 1766.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1777.30 | 1761.90 | 1767.13 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 1777.40 | 1770.81 | 1770.40 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 1759.60 | 1769.02 | 1770.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 1756.00 | 1762.54 | 1766.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 1771.20 | 1762.40 | 1763.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 10:15:00 | 1771.20 | 1762.40 | 1763.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1771.20 | 1762.40 | 1763.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 1771.20 | 1762.40 | 1763.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 1770.00 | 1763.92 | 1764.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:30:00 | 1770.30 | 1763.92 | 1764.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 1763.90 | 1762.81 | 1763.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:45:00 | 1766.20 | 1762.81 | 1763.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 1754.00 | 1761.05 | 1762.81 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2025-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 11:15:00 | 1779.80 | 1764.35 | 1763.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 1787.90 | 1773.93 | 1769.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 13:15:00 | 1773.70 | 1780.09 | 1774.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 13:15:00 | 1773.70 | 1780.09 | 1774.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 1773.70 | 1780.09 | 1774.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 1777.00 | 1780.09 | 1774.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 1781.00 | 1780.27 | 1774.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 15:15:00 | 1792.00 | 1780.27 | 1774.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 10:15:00 | 1785.80 | 1782.85 | 1777.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 10:15:00 | 1784.50 | 1822.58 | 1826.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 10:15:00 | 1784.50 | 1822.58 | 1826.55 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2025-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 15:15:00 | 1810.00 | 1805.98 | 1805.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 09:15:00 | 1815.60 | 1807.90 | 1806.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 12:15:00 | 1805.20 | 1809.01 | 1807.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 12:15:00 | 1805.20 | 1809.01 | 1807.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 1805.20 | 1809.01 | 1807.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 13:00:00 | 1805.20 | 1809.01 | 1807.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 1805.70 | 1808.34 | 1807.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 13:30:00 | 1799.40 | 1808.34 | 1807.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1805.80 | 1809.45 | 1808.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:30:00 | 1802.80 | 1809.45 | 1808.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 1806.50 | 1808.86 | 1808.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:45:00 | 1800.90 | 1808.86 | 1808.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 12:15:00 | 1803.00 | 1806.78 | 1807.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 13:15:00 | 1801.20 | 1805.67 | 1806.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 15:15:00 | 1808.50 | 1805.54 | 1806.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 15:15:00 | 1808.50 | 1805.54 | 1806.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 1808.50 | 1805.54 | 1806.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 09:15:00 | 1789.80 | 1805.54 | 1806.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 1792.00 | 1785.44 | 1785.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 1792.00 | 1785.44 | 1785.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 10:15:00 | 1793.00 | 1786.95 | 1786.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 09:15:00 | 1795.10 | 1803.62 | 1798.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 09:15:00 | 1795.10 | 1803.62 | 1798.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1795.10 | 1803.62 | 1798.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:00:00 | 1795.10 | 1803.62 | 1798.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 1811.90 | 1805.28 | 1799.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:30:00 | 1810.00 | 1805.28 | 1799.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 1806.10 | 1805.44 | 1800.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:45:00 | 1811.40 | 1807.80 | 1803.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 12:15:00 | 1797.70 | 1805.77 | 1803.69 | SL hit (close<static) qty=1.00 sl=1798.20 alert=retest2 |

### Cycle 45 — SELL (started 2025-11-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 14:15:00 | 1789.00 | 1800.57 | 1801.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 15:15:00 | 1784.00 | 1797.25 | 1799.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 14:15:00 | 1699.10 | 1698.60 | 1711.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 14:30:00 | 1698.30 | 1698.60 | 1711.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 1695.30 | 1698.38 | 1704.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 1686.60 | 1696.60 | 1700.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 13:15:00 | 1689.10 | 1691.68 | 1696.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 14:00:00 | 1689.70 | 1691.28 | 1695.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 09:45:00 | 1678.10 | 1688.24 | 1693.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1690.80 | 1680.27 | 1685.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:45:00 | 1695.60 | 1680.27 | 1685.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 1690.00 | 1682.21 | 1685.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:45:00 | 1691.00 | 1682.21 | 1685.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-02 12:15:00 | 1701.90 | 1687.72 | 1687.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 12:15:00 | 1701.90 | 1687.72 | 1687.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 13:15:00 | 1715.40 | 1693.25 | 1690.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 1700.30 | 1702.43 | 1695.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-03 09:30:00 | 1699.60 | 1702.43 | 1695.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 1700.00 | 1701.94 | 1696.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:00:00 | 1700.00 | 1701.94 | 1696.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 1705.80 | 1704.21 | 1699.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 14:30:00 | 1705.30 | 1704.21 | 1699.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 1730.90 | 1709.85 | 1702.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 10:15:00 | 1735.10 | 1709.85 | 1702.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 15:15:00 | 1689.20 | 1701.92 | 1702.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2025-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 15:15:00 | 1689.20 | 1701.92 | 1702.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 09:15:00 | 1672.10 | 1695.96 | 1699.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 13:15:00 | 1663.00 | 1657.35 | 1669.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 14:00:00 | 1663.00 | 1657.35 | 1669.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 1664.20 | 1658.81 | 1667.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:15:00 | 1665.50 | 1658.81 | 1667.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 1663.00 | 1659.65 | 1666.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 12:30:00 | 1658.50 | 1659.74 | 1665.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 13:30:00 | 1658.00 | 1659.33 | 1664.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 14:00:00 | 1657.70 | 1659.33 | 1664.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 09:15:00 | 1653.30 | 1660.84 | 1664.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 1647.20 | 1658.11 | 1662.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:15:00 | 1637.80 | 1650.13 | 1656.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:45:00 | 1637.00 | 1648.13 | 1654.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 09:15:00 | 1638.40 | 1647.78 | 1651.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 09:45:00 | 1637.00 | 1645.48 | 1650.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-19 10:15:00 | 1575.57 | 1588.17 | 1601.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-19 10:15:00 | 1575.10 | 1588.17 | 1601.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-19 10:15:00 | 1574.82 | 1588.17 | 1601.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 1602.40 | 1586.15 | 1595.68 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 1602.40 | 1586.15 | 1595.68 | SL hit (close>ema200) qty=0.50 sl=1586.15 alert=retest2 |

### Cycle 48 — BUY (started 2025-12-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 12:15:00 | 1611.70 | 1601.45 | 1600.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 14:15:00 | 1617.70 | 1606.82 | 1603.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 15:15:00 | 1615.10 | 1621.19 | 1616.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 15:15:00 | 1615.10 | 1621.19 | 1616.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 1615.10 | 1621.19 | 1616.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 1612.40 | 1621.19 | 1616.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1607.90 | 1618.53 | 1616.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:00:00 | 1607.90 | 1618.53 | 1616.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 1608.00 | 1616.43 | 1615.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 12:00:00 | 1620.00 | 1617.14 | 1615.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 11:15:00 | 1611.00 | 1616.49 | 1616.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 1611.00 | 1616.49 | 1616.52 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 15:15:00 | 1620.20 | 1616.67 | 1616.41 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 09:15:00 | 1613.00 | 1615.94 | 1616.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 12:15:00 | 1605.40 | 1611.60 | 1613.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 1616.60 | 1611.43 | 1613.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 15:15:00 | 1616.60 | 1611.43 | 1613.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1616.60 | 1611.43 | 1613.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 09:45:00 | 1600.00 | 1611.39 | 1612.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 13:15:00 | 1620.00 | 1614.30 | 1613.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 1620.00 | 1614.30 | 1613.94 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 14:15:00 | 1607.00 | 1613.23 | 1614.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 09:15:00 | 1596.80 | 1609.39 | 1612.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 15:15:00 | 1604.10 | 1599.34 | 1604.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 15:15:00 | 1604.10 | 1599.34 | 1604.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 15:15:00 | 1604.10 | 1599.34 | 1604.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:45:00 | 1607.90 | 1600.07 | 1604.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 1603.00 | 1600.66 | 1604.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 13:45:00 | 1593.80 | 1599.96 | 1603.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1514.11 | 1541.43 | 1554.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 09:15:00 | 1526.80 | 1523.73 | 1536.75 | SL hit (close>ema200) qty=0.50 sl=1523.73 alert=retest2 |

### Cycle 54 — BUY (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 10:15:00 | 1444.90 | 1436.98 | 1436.20 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2026-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 14:15:00 | 1426.50 | 1436.29 | 1436.43 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 1476.20 | 1444.05 | 1439.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 1492.00 | 1466.04 | 1459.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1492.60 | 1504.53 | 1493.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 1492.60 | 1504.53 | 1493.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 1492.60 | 1504.53 | 1493.90 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 15:15:00 | 1486.00 | 1489.84 | 1490.06 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 09:15:00 | 1511.90 | 1494.25 | 1492.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 11:15:00 | 1517.30 | 1501.76 | 1496.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-09 09:15:00 | 1511.00 | 1515.94 | 1506.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 1511.00 | 1515.94 | 1506.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 1511.00 | 1515.94 | 1506.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:45:00 | 1506.10 | 1515.94 | 1506.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 1598.00 | 1615.12 | 1602.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 1574.80 | 1615.12 | 1602.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 1640.60 | 1620.21 | 1605.71 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 09:15:00 | 1600.00 | 1609.62 | 1610.31 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 11:15:00 | 1617.00 | 1608.17 | 1607.83 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 1602.00 | 1608.21 | 1608.70 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 1613.60 | 1609.31 | 1608.72 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2026-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 15:15:00 | 1604.70 | 1608.55 | 1608.69 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 1616.50 | 1610.14 | 1609.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 10:15:00 | 1623.60 | 1612.83 | 1610.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 09:15:00 | 1605.30 | 1619.35 | 1615.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 09:15:00 | 1605.30 | 1619.35 | 1615.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 1605.30 | 1619.35 | 1615.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 1603.70 | 1619.35 | 1615.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 1600.00 | 1615.48 | 1614.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 10:30:00 | 1600.00 | 1615.48 | 1614.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 1604.10 | 1613.21 | 1613.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 14:15:00 | 1596.60 | 1606.48 | 1610.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 10:15:00 | 1594.50 | 1591.87 | 1597.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 10:15:00 | 1594.50 | 1591.87 | 1597.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 1594.50 | 1591.87 | 1597.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:45:00 | 1595.20 | 1591.87 | 1597.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 1597.00 | 1591.03 | 1594.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 1590.60 | 1591.03 | 1594.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1587.50 | 1590.32 | 1594.07 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 14:15:00 | 1605.20 | 1596.54 | 1595.78 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 1582.30 | 1593.45 | 1594.49 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2026-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 12:15:00 | 1597.30 | 1592.61 | 1592.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-04 13:15:00 | 1605.00 | 1595.09 | 1593.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 1704.40 | 1705.49 | 1667.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 1696.70 | 1712.29 | 1691.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 1696.70 | 1712.29 | 1691.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:30:00 | 1682.00 | 1712.29 | 1691.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 13:15:00 | 1697.30 | 1708.43 | 1695.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 13:45:00 | 1695.90 | 1708.43 | 1695.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 14:15:00 | 1698.00 | 1706.35 | 1696.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:30:00 | 1692.40 | 1706.35 | 1696.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 1694.20 | 1703.92 | 1695.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 1686.10 | 1703.92 | 1695.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 1690.20 | 1701.17 | 1695.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:30:00 | 1682.40 | 1701.17 | 1695.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 1687.50 | 1698.44 | 1694.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:00:00 | 1687.50 | 1698.44 | 1694.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 12:15:00 | 1665.10 | 1689.53 | 1691.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 1655.00 | 1679.08 | 1685.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 12:15:00 | 1671.10 | 1657.71 | 1670.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 12:15:00 | 1671.10 | 1657.71 | 1670.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 1671.10 | 1657.71 | 1670.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:00:00 | 1671.10 | 1657.71 | 1670.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 1659.50 | 1658.07 | 1669.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:30:00 | 1679.20 | 1658.07 | 1669.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 1655.00 | 1637.26 | 1648.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 09:15:00 | 1593.90 | 1637.26 | 1648.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:45:00 | 1621.10 | 1625.81 | 1631.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 1540.04 | 1573.38 | 1587.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 1560.70 | 1556.88 | 1568.00 | SL hit (close>ema200) qty=0.50 sl=1556.88 alert=retest2 |

### Cycle 70 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1620.40 | 1581.64 | 1576.67 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 1560.30 | 1588.06 | 1589.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1541.50 | 1575.03 | 1582.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1585.30 | 1559.59 | 1568.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1585.30 | 1559.59 | 1568.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1585.30 | 1559.59 | 1568.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 1589.80 | 1559.59 | 1568.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1572.60 | 1562.19 | 1568.89 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 1579.10 | 1572.31 | 1572.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 15:15:00 | 1590.10 | 1575.87 | 1573.79 | Break + close above crossover candle high |

### Cycle 73 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 1493.30 | 1559.36 | 1566.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 11:15:00 | 1481.80 | 1532.73 | 1552.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 14:15:00 | 1480.70 | 1477.15 | 1502.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-06 15:00:00 | 1480.70 | 1477.15 | 1502.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1506.20 | 1482.02 | 1490.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:30:00 | 1510.00 | 1482.02 | 1490.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 1516.00 | 1488.82 | 1493.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:00:00 | 1516.00 | 1488.82 | 1493.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 15:15:00 | 1486.00 | 1492.93 | 1494.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:15:00 | 1479.00 | 1492.93 | 1494.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 10:15:00 | 1482.00 | 1467.47 | 1466.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 1482.00 | 1467.47 | 1466.33 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 10:15:00 | 1460.90 | 1465.91 | 1466.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-16 11:15:00 | 1452.20 | 1463.16 | 1465.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-16 15:15:00 | 1459.50 | 1459.28 | 1462.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-17 09:15:00 | 1460.50 | 1459.28 | 1462.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 76 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 1490.70 | 1465.56 | 1464.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 11:15:00 | 1509.10 | 1475.62 | 1469.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 15:15:00 | 1515.00 | 1516.77 | 1500.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 09:15:00 | 1510.20 | 1516.77 | 1500.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 1506.50 | 1515.60 | 1503.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:45:00 | 1507.00 | 1515.60 | 1503.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 1502.00 | 1512.88 | 1502.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 11:45:00 | 1501.50 | 1512.88 | 1502.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 1504.80 | 1511.26 | 1503.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 1510.00 | 1505.91 | 1502.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 10:30:00 | 1509.30 | 1505.54 | 1502.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 12:15:00 | 1507.00 | 1505.51 | 1503.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 13:00:00 | 1507.30 | 1505.87 | 1503.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 14:15:00 | 1501.60 | 1504.89 | 1503.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 15:00:00 | 1501.60 | 1504.89 | 1503.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 15:15:00 | 1500.00 | 1503.91 | 1503.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:15:00 | 1488.00 | 1503.91 | 1503.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 1493.40 | 1501.81 | 1502.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 1493.40 | 1501.81 | 1502.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 1482.50 | 1493.12 | 1497.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1487.80 | 1480.14 | 1485.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1487.80 | 1480.14 | 1485.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1487.80 | 1480.14 | 1485.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1489.70 | 1480.14 | 1485.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1490.90 | 1482.29 | 1486.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:15:00 | 1483.00 | 1482.29 | 1486.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:00:00 | 1487.20 | 1483.98 | 1486.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:45:00 | 1486.90 | 1482.98 | 1485.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 09:15:00 | 1483.60 | 1483.80 | 1485.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 1483.20 | 1483.68 | 1485.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 15:15:00 | 1474.00 | 1482.39 | 1484.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:45:00 | 1472.40 | 1480.55 | 1482.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:15:00 | 1408.85 | 1443.84 | 1453.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:15:00 | 1412.84 | 1443.84 | 1453.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:15:00 | 1412.56 | 1443.84 | 1453.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:15:00 | 1409.42 | 1443.84 | 1453.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:15:00 | 1400.30 | 1443.84 | 1453.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:15:00 | 1398.78 | 1443.84 | 1453.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 1441.50 | 1424.24 | 1435.47 | SL hit (close>ema200) qty=0.50 sl=1424.24 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-15 11:00:00 | 2058.00 | 2025-05-26 10:15:00 | 2039.50 | STOP_HIT | 1.00 | 0.90% |
| SELL | retest2 | 2025-05-15 15:15:00 | 2054.00 | 2025-05-26 10:15:00 | 2039.50 | STOP_HIT | 1.00 | 0.71% |
| SELL | retest2 | 2025-05-16 09:45:00 | 2060.50 | 2025-05-26 10:15:00 | 2039.50 | STOP_HIT | 1.00 | 1.02% |
| SELL | retest2 | 2025-05-19 11:45:00 | 2060.00 | 2025-05-26 10:15:00 | 2039.50 | STOP_HIT | 1.00 | 1.00% |
| SELL | retest2 | 2025-05-19 14:15:00 | 2049.10 | 2025-05-26 10:15:00 | 2039.50 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2025-05-19 15:15:00 | 2045.00 | 2025-05-26 10:15:00 | 2039.50 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2025-05-20 10:15:00 | 2047.30 | 2025-05-26 10:15:00 | 2039.50 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2025-05-20 11:15:00 | 2048.40 | 2025-05-26 10:15:00 | 2039.50 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2025-05-21 12:45:00 | 2027.50 | 2025-05-26 10:15:00 | 2039.50 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-05-22 09:15:00 | 2013.70 | 2025-05-26 10:15:00 | 2039.50 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-06-06 10:30:00 | 2032.30 | 2025-06-11 15:15:00 | 2055.10 | STOP_HIT | 1.00 | 1.12% |
| SELL | retest2 | 2025-06-16 09:30:00 | 2034.90 | 2025-06-19 12:15:00 | 1933.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-16 09:30:00 | 2034.90 | 2025-06-19 15:15:00 | 1963.00 | STOP_HIT | 0.50 | 3.53% |
| BUY | retest2 | 2025-06-30 09:15:00 | 1976.00 | 2025-06-30 12:15:00 | 1949.90 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-07-01 09:15:00 | 1976.80 | 2025-07-01 12:15:00 | 1947.20 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-07-18 15:15:00 | 2009.50 | 2025-07-23 14:15:00 | 2017.90 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2025-07-21 09:45:00 | 2010.60 | 2025-07-23 14:15:00 | 2017.90 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2025-07-21 12:45:00 | 2019.00 | 2025-07-23 14:15:00 | 2017.90 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2025-08-06 11:45:00 | 1919.20 | 2025-08-07 14:15:00 | 1935.30 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-08-06 14:45:00 | 1920.10 | 2025-08-07 14:15:00 | 1935.30 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-08-07 09:45:00 | 1919.00 | 2025-08-07 14:15:00 | 1935.30 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-09-01 09:30:00 | 1822.00 | 2025-09-03 09:15:00 | 1847.60 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-09-02 09:45:00 | 1821.10 | 2025-09-03 09:15:00 | 1847.60 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-09-08 13:00:00 | 1814.20 | 2025-09-16 09:15:00 | 1832.90 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-09-09 10:00:00 | 1809.20 | 2025-09-16 09:15:00 | 1832.90 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-09-10 10:00:00 | 1810.30 | 2025-09-16 09:15:00 | 1832.90 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-10-03 09:15:00 | 1815.80 | 2025-10-06 10:15:00 | 1788.80 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-10-17 15:15:00 | 1792.00 | 2025-10-30 10:15:00 | 1784.50 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-10-20 10:15:00 | 1785.80 | 2025-10-30 10:15:00 | 1784.50 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-11-07 09:15:00 | 1789.80 | 2025-11-12 09:15:00 | 1792.00 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2025-11-17 09:45:00 | 1811.40 | 2025-11-17 12:15:00 | 1797.70 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-11-28 09:15:00 | 1686.60 | 2025-12-02 12:15:00 | 1701.90 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-11-28 13:15:00 | 1689.10 | 2025-12-02 12:15:00 | 1701.90 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-11-28 14:00:00 | 1689.70 | 2025-12-02 12:15:00 | 1701.90 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-12-01 09:45:00 | 1678.10 | 2025-12-02 12:15:00 | 1701.90 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-12-04 10:15:00 | 1735.10 | 2025-12-04 15:15:00 | 1689.20 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-12-10 12:30:00 | 1658.50 | 2025-12-19 10:15:00 | 1575.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-10 13:30:00 | 1658.00 | 2025-12-19 10:15:00 | 1575.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-10 14:00:00 | 1657.70 | 2025-12-19 10:15:00 | 1574.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-10 12:30:00 | 1658.50 | 2025-12-19 14:15:00 | 1602.40 | STOP_HIT | 0.50 | 3.38% |
| SELL | retest2 | 2025-12-10 13:30:00 | 1658.00 | 2025-12-19 14:15:00 | 1602.40 | STOP_HIT | 0.50 | 3.35% |
| SELL | retest2 | 2025-12-10 14:00:00 | 1657.70 | 2025-12-19 14:15:00 | 1602.40 | STOP_HIT | 0.50 | 3.34% |
| SELL | retest2 | 2025-12-11 09:15:00 | 1653.30 | 2025-12-22 12:15:00 | 1611.70 | STOP_HIT | 1.00 | 2.52% |
| SELL | retest2 | 2025-12-12 10:15:00 | 1637.80 | 2025-12-22 12:15:00 | 1611.70 | STOP_HIT | 1.00 | 1.59% |
| SELL | retest2 | 2025-12-12 10:45:00 | 1637.00 | 2025-12-22 12:15:00 | 1611.70 | STOP_HIT | 1.00 | 1.55% |
| SELL | retest2 | 2025-12-15 09:15:00 | 1638.40 | 2025-12-22 12:15:00 | 1611.70 | STOP_HIT | 1.00 | 1.63% |
| SELL | retest2 | 2025-12-15 09:45:00 | 1637.00 | 2025-12-22 12:15:00 | 1611.70 | STOP_HIT | 1.00 | 1.55% |
| BUY | retest2 | 2025-12-26 12:00:00 | 1620.00 | 2025-12-29 11:15:00 | 1611.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-12-31 09:45:00 | 1600.00 | 2025-12-31 13:15:00 | 1620.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2026-01-05 13:45:00 | 1593.80 | 2026-01-12 09:15:00 | 1514.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 13:45:00 | 1593.80 | 2026-01-13 09:15:00 | 1526.80 | STOP_HIT | 0.50 | 4.20% |
| SELL | retest2 | 2026-03-16 09:15:00 | 1593.90 | 2026-03-23 09:15:00 | 1540.04 | PARTIAL | 0.50 | 3.38% |
| SELL | retest2 | 2026-03-16 09:15:00 | 1593.90 | 2026-03-24 11:15:00 | 1560.70 | STOP_HIT | 0.50 | 2.08% |
| SELL | retest2 | 2026-03-17 11:45:00 | 1621.10 | 2026-03-25 09:15:00 | 1620.40 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2026-04-09 09:15:00 | 1479.00 | 2026-04-15 10:15:00 | 1482.00 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2026-04-22 09:15:00 | 1510.00 | 2026-04-23 09:15:00 | 1493.40 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2026-04-22 10:30:00 | 1509.30 | 2026-04-23 09:15:00 | 1493.40 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-04-22 12:15:00 | 1507.00 | 2026-04-23 09:15:00 | 1493.40 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2026-04-22 13:00:00 | 1507.30 | 2026-04-23 09:15:00 | 1493.40 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-04-27 11:15:00 | 1483.00 | 2026-05-06 09:15:00 | 1408.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-27 13:00:00 | 1487.20 | 2026-05-06 09:15:00 | 1412.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-27 13:45:00 | 1486.90 | 2026-05-06 09:15:00 | 1412.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-28 09:15:00 | 1483.60 | 2026-05-06 09:15:00 | 1409.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-28 15:15:00 | 1474.00 | 2026-05-06 09:15:00 | 1400.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-29 13:45:00 | 1472.40 | 2026-05-06 09:15:00 | 1398.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-27 11:15:00 | 1483.00 | 2026-05-07 09:15:00 | 1441.50 | STOP_HIT | 0.50 | 2.80% |
| SELL | retest2 | 2026-04-27 13:00:00 | 1487.20 | 2026-05-07 09:15:00 | 1441.50 | STOP_HIT | 0.50 | 3.07% |
| SELL | retest2 | 2026-04-27 13:45:00 | 1486.90 | 2026-05-07 09:15:00 | 1441.50 | STOP_HIT | 0.50 | 3.05% |
| SELL | retest2 | 2026-04-28 09:15:00 | 1483.60 | 2026-05-07 09:15:00 | 1441.50 | STOP_HIT | 0.50 | 2.84% |
| SELL | retest2 | 2026-04-28 15:15:00 | 1474.00 | 2026-05-07 09:15:00 | 1441.50 | STOP_HIT | 0.50 | 2.20% |
| SELL | retest2 | 2026-04-29 13:45:00 | 1472.40 | 2026-05-07 09:15:00 | 1441.50 | STOP_HIT | 0.50 | 2.10% |
