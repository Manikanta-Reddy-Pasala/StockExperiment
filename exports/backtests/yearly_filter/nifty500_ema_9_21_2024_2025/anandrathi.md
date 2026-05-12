# Anand Rathi Wealth Ltd. (ANANDRATHI)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 3602.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 142 |
| ALERT1 | 90 |
| ALERT2 | 87 |
| ALERT2_SKIP | 51 |
| ALERT3 | 262 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 163 |
| PARTIAL | 7 |
| TARGET_HIT | 7 |
| STOP_HIT | 161 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 175 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 84 / 91
- **Target hits / Stop hits / Partials:** 7 / 161 / 7
- **Avg / median % per leg:** 0.74% / -0.14%
- **Sum % (uncompounded):** 129.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 110 | 54 | 49.1% | 4 | 106 | 0 | 0.97% | 106.4% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 0 | 5 | 0 | 0.72% | 3.6% |
| BUY @ 3rd Alert (retest2) | 105 | 51 | 48.6% | 4 | 101 | 0 | 0.98% | 102.7% |
| SELL (all) | 65 | 30 | 46.2% | 3 | 55 | 7 | 0.36% | 23.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 65 | 30 | 46.2% | 3 | 55 | 7 | 0.36% | 23.1% |
| retest1 (combined) | 5 | 3 | 60.0% | 0 | 5 | 0 | 0.72% | 3.6% |
| retest2 (combined) | 170 | 81 | 47.6% | 7 | 156 | 7 | 0.74% | 125.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 14:15:00 | 1976.20 | 1970.40 | 1970.25 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 15:15:00 | 1968.50 | 1970.02 | 1970.09 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 1977.50 | 1971.51 | 1970.76 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 10:15:00 | 1962.50 | 1969.71 | 1970.01 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 11:15:00 | 1975.00 | 1970.77 | 1970.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 12:15:00 | 1976.05 | 1971.83 | 1970.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 13:15:00 | 1971.80 | 1971.82 | 1971.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 13:15:00 | 1971.80 | 1971.82 | 1971.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 13:15:00 | 1971.80 | 1971.82 | 1971.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 13:30:00 | 1972.50 | 1971.82 | 1971.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 14:15:00 | 1973.18 | 1972.09 | 1971.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 15:00:00 | 1973.18 | 1972.09 | 1971.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 1965.00 | 1970.67 | 1970.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 10:00:00 | 1975.50 | 1971.64 | 1971.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 10:30:00 | 1975.25 | 1972.31 | 1971.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 12:15:00 | 1975.25 | 1972.66 | 1971.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 14:00:00 | 1979.98 | 1974.41 | 1972.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 1974.50 | 1974.43 | 1972.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 15:15:00 | 1994.58 | 1974.43 | 1972.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 1994.58 | 1978.46 | 1974.82 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-18 12:15:00 | 1972.50 | 1975.50 | 1975.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-05-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-18 12:15:00 | 1972.50 | 1975.50 | 1975.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 09:15:00 | 1968.53 | 1974.10 | 1975.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-21 13:15:00 | 1969.00 | 1966.38 | 1970.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 13:15:00 | 1969.00 | 1966.38 | 1970.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 13:15:00 | 1969.00 | 1966.38 | 1970.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-21 14:00:00 | 1969.00 | 1966.38 | 1970.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 1975.03 | 1968.11 | 1970.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-21 15:00:00 | 1975.03 | 1968.11 | 1970.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 1975.00 | 1969.49 | 1971.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:15:00 | 1970.05 | 1969.49 | 1971.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 1975.00 | 1970.59 | 1971.53 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 10:15:00 | 1982.50 | 1972.97 | 1972.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 12:15:00 | 1992.48 | 1977.44 | 1974.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 09:15:00 | 2027.50 | 2029.73 | 2013.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-24 09:30:00 | 2024.95 | 2029.73 | 2013.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 2035.58 | 2037.42 | 2030.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 12:30:00 | 2058.35 | 2046.13 | 2036.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 10:30:00 | 2055.18 | 2068.22 | 2066.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-31 12:15:00 | 2052.48 | 2063.14 | 2064.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-05-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 12:15:00 | 2052.48 | 2063.14 | 2064.14 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 14:15:00 | 2100.03 | 2069.94 | 2067.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 15:15:00 | 2121.90 | 2080.33 | 2072.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-03 10:15:00 | 2083.80 | 2084.25 | 2075.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 10:15:00 | 2083.80 | 2084.25 | 2075.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 10:15:00 | 2083.80 | 2084.25 | 2075.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-03 10:45:00 | 2065.45 | 2084.25 | 2075.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 12:15:00 | 2073.23 | 2080.98 | 2075.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-03 13:00:00 | 2073.23 | 2080.98 | 2075.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 13:15:00 | 2068.05 | 2078.39 | 2074.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-03 14:00:00 | 2068.05 | 2078.39 | 2074.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 14:15:00 | 2051.25 | 2072.96 | 2072.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-03 14:30:00 | 2049.25 | 2072.96 | 2072.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-03 15:15:00 | 2047.50 | 2067.87 | 2070.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 09:15:00 | 1995.00 | 2053.30 | 2063.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 1984.03 | 1959.89 | 1994.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 11:00:00 | 1984.03 | 1959.89 | 1994.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 1990.80 | 1966.07 | 1994.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:30:00 | 1992.48 | 1966.07 | 1994.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 1994.73 | 1971.80 | 1994.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:00:00 | 1994.73 | 1971.80 | 1994.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 1992.50 | 1975.94 | 1994.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:45:00 | 2000.83 | 1975.94 | 1994.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 2001.45 | 1981.04 | 1994.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 15:00:00 | 2001.45 | 1981.04 | 1994.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 2000.00 | 1984.84 | 1995.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:15:00 | 2046.00 | 1984.84 | 1995.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 1999.50 | 1990.21 | 1996.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:30:00 | 2001.50 | 1990.21 | 1996.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 12:15:00 | 2000.00 | 1993.74 | 1996.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 12:30:00 | 2001.18 | 1993.74 | 1996.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 13:15:00 | 2000.90 | 1995.17 | 1997.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-07 10:00:00 | 1980.00 | 1993.53 | 1996.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-12 11:15:00 | 1963.50 | 1953.93 | 1953.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2024-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 11:15:00 | 1963.50 | 1953.93 | 1953.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-12 12:15:00 | 1974.95 | 1958.14 | 1955.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 10:15:00 | 1995.03 | 1999.74 | 1990.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-19 10:30:00 | 1998.73 | 1999.74 | 1990.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 14:15:00 | 2005.28 | 2003.03 | 1995.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 15:00:00 | 2005.28 | 2003.03 | 1995.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 15:15:00 | 1995.00 | 2001.43 | 1995.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:15:00 | 2004.53 | 2001.43 | 1995.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 2012.48 | 2003.64 | 1996.98 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2024-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 12:15:00 | 1992.50 | 1997.50 | 1997.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 13:15:00 | 1971.50 | 1992.30 | 1995.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 09:15:00 | 1933.93 | 1927.49 | 1939.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 09:15:00 | 1933.93 | 1927.49 | 1939.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 1933.93 | 1927.49 | 1939.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 10:15:00 | 1922.50 | 1927.49 | 1939.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 11:45:00 | 1924.70 | 1926.59 | 1937.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 12:15:00 | 1922.50 | 1926.59 | 1937.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 12:45:00 | 1922.50 | 1925.98 | 1935.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 1943.35 | 1931.06 | 1935.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:15:00 | 1961.40 | 1931.06 | 1935.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 1968.23 | 1938.50 | 1938.81 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-06-28 09:15:00 | 1968.23 | 1938.50 | 1938.81 | SL hit (close>static) qty=1.00 sl=1955.25 alert=retest2 |

### Cycle 13 — BUY (started 2024-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 10:15:00 | 1968.53 | 1944.50 | 1941.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 11:15:00 | 1993.00 | 1954.20 | 1946.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 14:15:00 | 1945.50 | 1961.72 | 1952.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 14:15:00 | 1945.50 | 1961.72 | 1952.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 1945.50 | 1961.72 | 1952.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 15:00:00 | 1945.50 | 1961.72 | 1952.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 1991.88 | 1967.76 | 1955.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:45:00 | 1945.00 | 1961.65 | 1954.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 1946.98 | 1958.72 | 1953.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:30:00 | 1937.50 | 1958.72 | 1953.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 12:15:00 | 1952.85 | 1957.14 | 1953.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 13:00:00 | 1952.85 | 1957.14 | 1953.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 13:15:00 | 1949.43 | 1955.60 | 1953.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 13:30:00 | 1950.73 | 1955.60 | 1953.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 14:15:00 | 1962.48 | 1956.97 | 1954.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 15:15:00 | 1962.53 | 1956.97 | 1954.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 10:15:00 | 1943.63 | 1953.80 | 1953.46 | SL hit (close<static) qty=1.00 sl=1947.50 alert=retest2 |

### Cycle 14 — SELL (started 2024-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 11:15:00 | 1937.90 | 1950.62 | 1952.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 12:15:00 | 1932.50 | 1947.00 | 1950.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 10:15:00 | 1943.00 | 1941.95 | 1946.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 10:15:00 | 1943.00 | 1941.95 | 1946.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 1943.00 | 1941.95 | 1946.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 10:45:00 | 1950.70 | 1941.95 | 1946.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 11:15:00 | 1946.95 | 1942.95 | 1946.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 11:30:00 | 1946.18 | 1942.95 | 1946.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 12:15:00 | 1935.00 | 1941.36 | 1945.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 10:30:00 | 1932.50 | 1940.31 | 1943.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 11:30:00 | 1934.00 | 1940.45 | 1943.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-04 15:15:00 | 1950.00 | 1945.13 | 1944.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 15:15:00 | 1950.00 | 1945.13 | 1944.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 13:15:00 | 1964.95 | 1949.78 | 1947.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 2037.65 | 2046.66 | 2027.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-10 10:00:00 | 2037.65 | 2046.66 | 2027.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 2040.00 | 2045.33 | 2028.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 2007.65 | 2045.33 | 2028.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 2062.95 | 2048.85 | 2031.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 11:45:00 | 2035.50 | 2048.85 | 2031.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 2073.28 | 2067.16 | 2055.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:45:00 | 2064.35 | 2067.16 | 2055.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 11:15:00 | 2060.20 | 2067.58 | 2057.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 12:00:00 | 2060.20 | 2067.58 | 2057.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 12:15:00 | 2052.50 | 2064.56 | 2057.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 12:30:00 | 2052.50 | 2064.56 | 2057.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 13:15:00 | 2032.58 | 2058.16 | 2054.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 14:00:00 | 2032.58 | 2058.16 | 2054.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2024-07-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 15:15:00 | 1999.98 | 2044.53 | 2049.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 10:15:00 | 1915.50 | 1946.07 | 1971.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 14:15:00 | 1942.53 | 1936.42 | 1957.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-18 15:00:00 | 1942.53 | 1936.42 | 1957.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 1900.15 | 1882.71 | 1909.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:45:00 | 1916.40 | 1882.71 | 1909.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 1915.00 | 1891.10 | 1904.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 09:15:00 | 1897.00 | 1891.10 | 1904.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 1885.00 | 1898.15 | 1904.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 13:00:00 | 1890.38 | 1896.59 | 1903.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 14:15:00 | 1887.65 | 1896.76 | 1903.00 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 15:15:00 | 1905.50 | 1896.97 | 1901.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:15:00 | 1904.65 | 1896.97 | 1901.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 1913.50 | 1900.28 | 1902.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 1913.50 | 1900.28 | 1902.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 1906.15 | 1901.45 | 1903.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 15:00:00 | 1900.05 | 1901.80 | 1902.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-30 11:15:00 | 1880.53 | 1867.24 | 1866.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 11:15:00 | 1880.53 | 1867.24 | 1866.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 09:15:00 | 1895.23 | 1879.42 | 1873.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 13:15:00 | 1885.48 | 1885.81 | 1878.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 14:00:00 | 1885.48 | 1885.81 | 1878.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 1862.48 | 1881.14 | 1878.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:45:00 | 1863.50 | 1881.14 | 1878.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 1862.23 | 1877.36 | 1876.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 11:00:00 | 1862.23 | 1877.36 | 1876.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 11:15:00 | 1862.43 | 1874.37 | 1875.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 1850.00 | 1862.42 | 1868.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 13:15:00 | 1794.50 | 1793.58 | 1805.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 14:00:00 | 1794.50 | 1793.58 | 1805.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 1793.63 | 1793.05 | 1802.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 09:30:00 | 1798.08 | 1793.05 | 1802.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 10:15:00 | 1797.83 | 1794.01 | 1802.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 15:00:00 | 1792.15 | 1795.59 | 1800.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 09:15:00 | 1823.88 | 1799.96 | 1801.51 | SL hit (close>static) qty=1.00 sl=1802.38 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 10:15:00 | 1823.28 | 1804.62 | 1803.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 12:15:00 | 1835.00 | 1813.31 | 1807.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 14:15:00 | 1805.50 | 1816.11 | 1810.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 14:15:00 | 1805.50 | 1816.11 | 1810.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 1805.50 | 1816.11 | 1810.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 15:00:00 | 1805.50 | 1816.11 | 1810.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 1822.00 | 1817.28 | 1811.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:15:00 | 1843.55 | 1817.28 | 1811.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 1828.68 | 1819.56 | 1812.93 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2024-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 13:15:00 | 1792.73 | 1817.92 | 1818.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 14:15:00 | 1784.20 | 1811.17 | 1815.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 15:15:00 | 1798.50 | 1798.13 | 1804.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 09:15:00 | 1801.20 | 1798.13 | 1804.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 1799.98 | 1798.50 | 1804.23 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2024-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 09:15:00 | 1891.48 | 1816.87 | 1807.74 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 13:15:00 | 1816.48 | 1847.40 | 1849.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 14:15:00 | 1800.00 | 1837.92 | 1844.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 13:15:00 | 1822.58 | 1801.37 | 1819.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 13:15:00 | 1822.58 | 1801.37 | 1819.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 13:15:00 | 1822.58 | 1801.37 | 1819.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 13:45:00 | 1818.03 | 1801.37 | 1819.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 14:15:00 | 1820.00 | 1805.10 | 1819.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 15:15:00 | 1845.00 | 1805.10 | 1819.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 15:15:00 | 1845.00 | 1813.08 | 1821.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:15:00 | 1850.53 | 1813.08 | 1821.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 1857.50 | 1821.96 | 1824.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:00:00 | 1857.50 | 1821.96 | 1824.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2024-08-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 10:15:00 | 1850.98 | 1827.76 | 1827.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 11:15:00 | 1874.00 | 1837.01 | 1831.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 12:15:00 | 1861.50 | 1862.93 | 1851.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-28 13:00:00 | 1861.50 | 1862.93 | 1851.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 1873.78 | 1865.62 | 1856.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 12:15:00 | 1891.48 | 1874.25 | 1865.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 10:15:00 | 1910.85 | 1937.95 | 1940.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 10:15:00 | 1910.85 | 1937.95 | 1940.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 11:15:00 | 1904.00 | 1931.16 | 1937.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 1927.50 | 1922.87 | 1929.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 1927.50 | 1922.87 | 1929.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 1927.50 | 1922.87 | 1929.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:45:00 | 1937.70 | 1922.87 | 1929.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 1922.78 | 1922.85 | 1929.28 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2024-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 14:15:00 | 1942.00 | 1931.97 | 1931.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 09:15:00 | 1965.38 | 1939.24 | 1935.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 12:15:00 | 1953.05 | 1957.09 | 1950.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-12 12:45:00 | 1955.05 | 1957.09 | 1950.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 13:15:00 | 1952.03 | 1956.08 | 1950.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 13:30:00 | 1947.53 | 1956.08 | 1950.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 1950.40 | 1954.94 | 1950.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 15:00:00 | 1950.40 | 1954.94 | 1950.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 15:15:00 | 1947.50 | 1953.45 | 1950.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 09:45:00 | 1955.83 | 1953.26 | 1950.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 11:00:00 | 1952.85 | 1953.18 | 1950.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 12:15:00 | 1957.93 | 1967.84 | 1968.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 12:15:00 | 1957.93 | 1967.84 | 1968.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 14:15:00 | 1956.53 | 1964.07 | 1966.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 15:15:00 | 1950.00 | 1929.78 | 1943.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 15:15:00 | 1950.00 | 1929.78 | 1943.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 1950.00 | 1929.78 | 1943.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 11:30:00 | 1898.35 | 1917.97 | 1934.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 14:15:00 | 1955.40 | 1922.36 | 1931.69 | SL hit (close>static) qty=1.00 sl=1950.00 alert=retest2 |

### Cycle 27 — BUY (started 2024-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 13:15:00 | 1945.50 | 1934.48 | 1934.46 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 09:15:00 | 1923.60 | 1933.70 | 1934.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 13:15:00 | 1920.80 | 1930.85 | 1932.70 | Break + close below crossover candle low |

### Cycle 29 — BUY (started 2024-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 14:15:00 | 1959.03 | 1936.48 | 1935.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-25 11:15:00 | 1971.93 | 1951.13 | 1943.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 13:15:00 | 1949.80 | 1953.46 | 1945.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 13:15:00 | 1949.80 | 1953.46 | 1945.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 1949.80 | 1953.46 | 1945.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 13:45:00 | 1946.30 | 1953.46 | 1945.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 1945.00 | 1951.64 | 1946.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 09:15:00 | 1967.30 | 1951.64 | 1946.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 14:30:00 | 1959.48 | 1957.29 | 1952.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 09:15:00 | 1955.58 | 1956.20 | 1952.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 12:30:00 | 1957.08 | 1957.24 | 1954.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 13:15:00 | 1950.90 | 1955.97 | 1954.04 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-27 14:15:00 | 1930.90 | 1950.96 | 1951.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 14:15:00 | 1930.90 | 1950.96 | 1951.94 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 11:15:00 | 1982.13 | 1950.93 | 1950.74 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 11:15:00 | 1941.60 | 1952.37 | 1953.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 10:15:00 | 1927.08 | 1945.78 | 1949.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-03 14:15:00 | 1953.55 | 1943.02 | 1946.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 14:15:00 | 1953.55 | 1943.02 | 1946.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 1953.55 | 1943.02 | 1946.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-03 15:00:00 | 1953.55 | 1943.02 | 1946.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 15:15:00 | 1952.50 | 1944.92 | 1947.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 09:15:00 | 1921.25 | 1944.92 | 1947.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 10:15:00 | 1952.80 | 1949.25 | 1948.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 10:15:00 | 1952.80 | 1949.25 | 1948.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-04 11:15:00 | 1978.00 | 1955.00 | 1951.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 14:15:00 | 1957.40 | 1958.30 | 1954.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 14:15:00 | 1957.40 | 1958.30 | 1954.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 1957.40 | 1958.30 | 1954.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 14:45:00 | 1950.00 | 1958.30 | 1954.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 15:15:00 | 1956.00 | 1957.84 | 1954.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 09:15:00 | 1976.40 | 1957.84 | 1954.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 1944.28 | 1955.13 | 1953.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 10:00:00 | 1944.28 | 1955.13 | 1953.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 10:15:00 | 1927.38 | 1949.58 | 1951.07 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 12:15:00 | 1965.93 | 1952.31 | 1950.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 13:15:00 | 1974.43 | 1956.73 | 1952.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 10:15:00 | 2037.28 | 2041.62 | 2013.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 10:30:00 | 2035.88 | 2041.62 | 2013.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 2028.50 | 2032.95 | 2019.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 09:15:00 | 2072.90 | 2032.95 | 2019.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 09:15:00 | 2054.60 | 2031.34 | 2025.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 09:30:00 | 2055.90 | 2030.15 | 2028.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 10:15:00 | 2054.53 | 2030.15 | 2028.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 2040.43 | 2033.80 | 2030.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:45:00 | 2042.18 | 2033.80 | 2030.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 2038.40 | 2034.72 | 2031.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 13:15:00 | 2050.00 | 2034.72 | 2031.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 13:45:00 | 2064.32 | 2040.02 | 2033.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 09:30:00 | 2059.07 | 2052.03 | 2041.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 10:00:00 | 2059.35 | 2052.03 | 2041.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 2040.48 | 2060.45 | 2053.33 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-17 14:15:00 | 2036.55 | 2047.28 | 2048.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 14:15:00 | 2036.55 | 2047.28 | 2048.66 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 09:15:00 | 2065.98 | 2050.65 | 2049.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 10:15:00 | 2080.63 | 2056.65 | 2052.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 10:15:00 | 2103.73 | 2114.66 | 2091.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 10:15:00 | 2103.73 | 2114.66 | 2091.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 2103.73 | 2114.66 | 2091.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 10:30:00 | 2101.50 | 2114.66 | 2091.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 2098.95 | 2111.52 | 2092.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 12:00:00 | 2098.95 | 2111.52 | 2092.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 12:15:00 | 2090.78 | 2107.37 | 2091.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 13:00:00 | 2090.78 | 2107.37 | 2091.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 13:15:00 | 2075.00 | 2100.89 | 2090.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 14:00:00 | 2075.00 | 2100.89 | 2090.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 14:15:00 | 2071.25 | 2094.97 | 2088.66 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2024-10-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 09:15:00 | 2059.55 | 2083.89 | 2084.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 11:15:00 | 2034.48 | 2067.65 | 2076.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 09:15:00 | 2060.88 | 2033.96 | 2044.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-24 09:15:00 | 2060.88 | 2033.96 | 2044.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 2060.88 | 2033.96 | 2044.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:00:00 | 2060.88 | 2033.96 | 2044.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 2055.45 | 2038.26 | 2045.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 11:30:00 | 2041.75 | 2038.78 | 2044.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 15:15:00 | 2043.98 | 2036.87 | 2042.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 15:15:00 | 1939.66 | 1967.43 | 1976.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 15:15:00 | 1941.78 | 1967.43 | 1976.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-31 09:15:00 | 1991.60 | 1972.26 | 1978.23 | SL hit (close>ema200) qty=0.50 sl=1972.26 alert=retest2 |

### Cycle 39 — BUY (started 2024-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 11:15:00 | 2020.95 | 1984.42 | 1982.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 12:15:00 | 2045.98 | 1996.73 | 1988.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 2006.40 | 2037.23 | 2017.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 2006.40 | 2037.23 | 2017.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 2006.40 | 2037.23 | 2017.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 2006.40 | 2037.23 | 2017.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 2005.98 | 2030.98 | 2016.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:30:00 | 2001.73 | 2030.98 | 2016.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 2012.70 | 2027.33 | 2015.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 13:45:00 | 2020.35 | 2023.18 | 2015.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 14:30:00 | 2019.60 | 2020.72 | 2015.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:30:00 | 2019.00 | 2021.17 | 2016.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 11:00:00 | 2022.18 | 2021.37 | 2016.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 11:15:00 | 2018.13 | 2020.73 | 2017.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 12:00:00 | 2018.13 | 2020.73 | 2017.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 12:15:00 | 2019.40 | 2020.46 | 2017.29 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-06 11:15:00 | 2002.53 | 2014.20 | 2015.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-06 11:15:00 | 2002.53 | 2014.20 | 2015.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-06 12:15:00 | 1994.10 | 2010.18 | 2013.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 09:15:00 | 1987.65 | 1972.40 | 1980.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 09:15:00 | 1987.65 | 1972.40 | 1980.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 1987.65 | 1972.40 | 1980.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:30:00 | 1968.73 | 1972.40 | 1980.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 2007.60 | 1979.44 | 1983.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 10:45:00 | 2007.00 | 1979.44 | 1983.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 1988.50 | 1981.25 | 1983.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:30:00 | 2007.28 | 1981.25 | 1983.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 14:15:00 | 1973.70 | 1980.90 | 1983.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 14:30:00 | 1977.75 | 1980.90 | 1983.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 1974.90 | 1978.36 | 1981.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:15:00 | 1959.73 | 1974.84 | 1979.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 14:45:00 | 1955.33 | 1966.14 | 1973.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 09:15:00 | 1920.00 | 1965.21 | 1972.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 11:15:00 | 1971.75 | 1950.35 | 1947.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2024-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 11:15:00 | 1971.75 | 1950.35 | 1947.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-18 12:15:00 | 1996.05 | 1959.49 | 1952.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-21 12:15:00 | 2014.48 | 2016.48 | 1998.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-21 13:00:00 | 2014.48 | 2016.48 | 1998.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 14:15:00 | 2007.50 | 2012.56 | 1999.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 15:00:00 | 2007.50 | 2012.56 | 1999.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 1986.05 | 2005.25 | 1998.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 09:45:00 | 1987.00 | 2005.25 | 1998.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 1977.83 | 1999.77 | 1996.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 11:00:00 | 1977.83 | 1999.77 | 1996.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 13:15:00 | 1990.48 | 1994.50 | 1994.77 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 2007.38 | 1996.33 | 1995.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 11:15:00 | 2019.75 | 2001.01 | 1997.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 11:15:00 | 2038.70 | 2038.93 | 2023.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 11:45:00 | 2034.63 | 2038.93 | 2023.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 12:15:00 | 2045.03 | 2048.18 | 2038.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 14:45:00 | 2062.07 | 2045.52 | 2041.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 15:15:00 | 2059.00 | 2045.52 | 2041.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 10:15:00 | 2058.48 | 2050.04 | 2044.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-09 10:15:00 | 2268.28 | 2196.49 | 2160.26 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 2168.73 | 2206.07 | 2208.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 2152.50 | 2174.42 | 2188.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 13:15:00 | 2106.00 | 2103.51 | 2118.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-18 14:00:00 | 2106.00 | 2103.51 | 2118.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 1991.05 | 1983.50 | 2001.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 09:15:00 | 1989.73 | 1996.79 | 1998.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 10:00:00 | 1988.95 | 1995.23 | 1998.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 12:15:00 | 1985.03 | 1993.50 | 1996.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 13:15:00 | 1990.00 | 1993.18 | 1996.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 1990.98 | 1983.63 | 1989.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 10:30:00 | 1990.10 | 1983.63 | 1989.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 11:15:00 | 1982.60 | 1983.42 | 1988.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 11:45:00 | 1984.80 | 1983.42 | 1988.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 1983.33 | 1978.67 | 1983.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 10:00:00 | 1983.33 | 1978.67 | 1983.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 1983.90 | 1979.72 | 1983.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 11:00:00 | 1983.90 | 1979.72 | 1983.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 11:15:00 | 1972.28 | 1978.23 | 1982.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 12:15:00 | 1968.38 | 1978.23 | 1982.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 12:45:00 | 1964.83 | 1975.76 | 1981.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 13:45:00 | 1967.50 | 1974.21 | 1980.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 14:30:00 | 1969.23 | 1972.87 | 1978.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 1962.50 | 1971.72 | 1977.39 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-03 14:15:00 | 1980.00 | 1975.87 | 1975.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 14:15:00 | 1980.00 | 1975.87 | 1975.47 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 1945.70 | 1970.93 | 1973.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 11:15:00 | 1940.93 | 1964.93 | 1970.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 1956.33 | 1946.24 | 1957.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 1956.33 | 1946.24 | 1957.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 1956.33 | 1946.24 | 1957.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 10:15:00 | 1927.75 | 1944.32 | 1950.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 09:15:00 | 1970.50 | 1930.75 | 1936.74 | SL hit (close>static) qty=1.00 sl=1969.75 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 11:15:00 | 1975.50 | 1946.44 | 1943.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-10 10:15:00 | 2028.50 | 1975.48 | 1960.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-13 09:15:00 | 2027.73 | 2038.84 | 2006.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-13 09:30:00 | 2034.50 | 2038.84 | 2006.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 12:15:00 | 2020.00 | 2030.50 | 2010.27 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 11:15:00 | 1983.00 | 1999.77 | 2001.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-15 11:15:00 | 1958.53 | 1978.95 | 1988.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 13:15:00 | 2016.73 | 1984.45 | 1989.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-15 13:15:00 | 2016.73 | 1984.45 | 1989.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 13:15:00 | 2016.73 | 1984.45 | 1989.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 14:00:00 | 2016.73 | 1984.45 | 1989.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 14:15:00 | 2002.23 | 1988.00 | 1990.66 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 2000.03 | 1993.06 | 1992.66 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 14:15:00 | 1984.05 | 1992.09 | 1992.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-16 15:15:00 | 1975.55 | 1988.78 | 1990.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-17 11:15:00 | 1993.43 | 1988.16 | 1989.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 11:15:00 | 1993.43 | 1988.16 | 1989.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 1993.43 | 1988.16 | 1989.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:30:00 | 1995.90 | 1988.16 | 1989.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 12:15:00 | 1991.15 | 1988.76 | 1990.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 13:00:00 | 1991.15 | 1988.76 | 1990.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 1987.60 | 1988.52 | 1989.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 13:30:00 | 1989.58 | 1988.52 | 1989.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 1989.50 | 1988.72 | 1989.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:45:00 | 1992.65 | 1988.72 | 1989.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 1980.08 | 1986.99 | 1988.92 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 11:15:00 | 1999.20 | 1990.78 | 1990.16 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 14:15:00 | 1978.65 | 1987.78 | 1988.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 09:15:00 | 1937.75 | 1976.05 | 1983.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 1915.53 | 1904.23 | 1927.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 1915.53 | 1904.23 | 1927.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 15:15:00 | 1780.00 | 1744.55 | 1775.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 09:15:00 | 1747.50 | 1744.55 | 1775.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 11:15:00 | 1751.23 | 1747.08 | 1771.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 09:15:00 | 1783.45 | 1758.14 | 1765.47 | SL hit (close>static) qty=1.00 sl=1782.98 alert=retest2 |

### Cycle 53 — BUY (started 2025-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 11:15:00 | 1853.25 | 1784.65 | 1776.72 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-02-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 15:15:00 | 1812.50 | 1816.16 | 1816.36 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 10:15:00 | 1827.05 | 1818.37 | 1817.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 1871.10 | 1834.95 | 1826.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 10:15:00 | 1884.33 | 1887.40 | 1865.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 09:15:00 | 1881.55 | 1883.48 | 1872.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1881.55 | 1883.48 | 1872.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:15:00 | 1856.18 | 1883.48 | 1872.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 1864.40 | 1879.66 | 1872.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:45:00 | 1856.78 | 1879.66 | 1872.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 1872.80 | 1878.29 | 1872.11 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2025-02-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 14:15:00 | 1850.00 | 1865.17 | 1867.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 1838.45 | 1854.63 | 1861.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 14:15:00 | 1852.18 | 1850.98 | 1857.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-10 15:00:00 | 1852.18 | 1850.98 | 1857.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 15:15:00 | 1850.00 | 1850.79 | 1856.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 09:30:00 | 1810.08 | 1845.39 | 1853.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-18 11:15:00 | 1829.40 | 1809.97 | 1807.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2025-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 11:15:00 | 1829.40 | 1809.97 | 1807.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 14:15:00 | 1851.90 | 1824.76 | 1815.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-25 10:15:00 | 1990.55 | 1992.74 | 1967.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-25 11:00:00 | 1990.55 | 1992.74 | 1967.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 11:15:00 | 1985.70 | 2000.82 | 1987.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 12:00:00 | 1985.70 | 2000.82 | 1987.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 12:15:00 | 1993.40 | 1999.33 | 1987.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 12:30:00 | 1980.90 | 1999.33 | 1987.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 13:15:00 | 1992.58 | 1997.98 | 1988.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 13:45:00 | 1993.00 | 1997.98 | 1988.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 1999.98 | 1998.38 | 1989.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 14:30:00 | 1989.58 | 1998.38 | 1989.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 1993.20 | 1998.08 | 1990.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-28 12:45:00 | 2004.70 | 1998.06 | 1992.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-28 13:15:00 | 2001.83 | 1998.06 | 1992.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-03 09:45:00 | 2001.53 | 2001.97 | 1996.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 1933.95 | 2031.48 | 2033.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-05 09:15:00 | 1933.95 | 2031.48 | 2033.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-05 10:15:00 | 1917.00 | 2008.59 | 2023.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-07 14:15:00 | 1865.10 | 1846.12 | 1881.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-07 14:45:00 | 1860.70 | 1846.12 | 1881.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 1861.10 | 1849.12 | 1879.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-10 09:15:00 | 1832.60 | 1849.12 | 1879.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-11 09:15:00 | 1740.97 | 1792.84 | 1828.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-11 10:15:00 | 1802.90 | 1794.85 | 1826.44 | SL hit (close>ema200) qty=0.50 sl=1794.85 alert=retest2 |

### Cycle 59 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 1722.60 | 1693.86 | 1693.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 10:15:00 | 1728.85 | 1700.86 | 1696.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-18 12:15:00 | 1700.05 | 1702.96 | 1698.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-18 13:00:00 | 1700.05 | 1702.96 | 1698.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 13:15:00 | 1698.30 | 1702.03 | 1698.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 13:45:00 | 1695.75 | 1702.03 | 1698.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 14:15:00 | 1703.05 | 1702.23 | 1699.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 15:15:00 | 1706.00 | 1702.23 | 1699.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 15:15:00 | 1706.00 | 1702.99 | 1699.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 09:15:00 | 1703.85 | 1702.99 | 1699.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 1686.80 | 1699.75 | 1698.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 10:00:00 | 1686.80 | 1699.75 | 1698.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2025-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-19 10:15:00 | 1686.40 | 1697.08 | 1697.43 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 15:15:00 | 1710.00 | 1698.37 | 1697.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 10:15:00 | 1724.70 | 1705.64 | 1700.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 10:15:00 | 1792.00 | 1792.80 | 1770.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:30:00 | 1795.20 | 1792.80 | 1770.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 1778.65 | 1786.73 | 1773.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 14:15:00 | 1814.15 | 1786.73 | 1773.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 15:15:00 | 1825.10 | 1842.50 | 1844.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 15:15:00 | 1825.10 | 1842.50 | 1844.25 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 09:15:00 | 1862.85 | 1846.57 | 1845.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 11:15:00 | 1869.20 | 1853.24 | 1849.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 09:15:00 | 1865.85 | 1867.56 | 1859.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 09:15:00 | 1865.85 | 1867.56 | 1859.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 1865.85 | 1867.56 | 1859.29 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2025-04-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 13:15:00 | 1838.00 | 1855.07 | 1855.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-03 14:15:00 | 1837.45 | 1851.54 | 1854.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 1804.90 | 1743.97 | 1774.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 15:15:00 | 1804.90 | 1743.97 | 1774.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 1804.90 | 1743.97 | 1774.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:15:00 | 1801.20 | 1743.97 | 1774.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1800.15 | 1755.21 | 1776.69 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2025-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 15:15:00 | 1796.00 | 1785.51 | 1785.25 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 09:15:00 | 1760.05 | 1780.42 | 1782.96 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 11:15:00 | 1806.00 | 1786.23 | 1785.21 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-04-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 15:15:00 | 1773.95 | 1783.60 | 1784.70 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-04-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 14:15:00 | 1814.00 | 1789.25 | 1786.16 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-15 13:15:00 | 1780.00 | 1785.82 | 1786.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-15 14:15:00 | 1773.50 | 1783.36 | 1784.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-17 11:15:00 | 1739.00 | 1736.71 | 1750.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-17 11:45:00 | 1738.80 | 1736.71 | 1750.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 1732.00 | 1734.22 | 1745.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:15:00 | 1730.10 | 1734.22 | 1745.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 1729.00 | 1733.18 | 1743.71 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2025-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 09:15:00 | 1766.20 | 1749.30 | 1747.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 10:15:00 | 1774.00 | 1754.24 | 1750.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 14:15:00 | 1761.00 | 1762.72 | 1756.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 15:00:00 | 1761.00 | 1762.72 | 1756.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 1766.50 | 1763.47 | 1757.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 10:15:00 | 1779.00 | 1765.24 | 1758.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 11:00:00 | 1781.90 | 1768.57 | 1760.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 09:30:00 | 1790.00 | 1784.39 | 1773.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 1730.00 | 1768.09 | 1770.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 1730.00 | 1768.09 | 1770.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 14:15:00 | 1700.00 | 1722.28 | 1733.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-30 13:15:00 | 1717.80 | 1701.32 | 1715.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 13:15:00 | 1717.80 | 1701.32 | 1715.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 13:15:00 | 1717.80 | 1701.32 | 1715.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 14:00:00 | 1717.80 | 1701.32 | 1715.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 14:15:00 | 1707.00 | 1702.46 | 1714.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 15:15:00 | 1692.00 | 1702.46 | 1714.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 12:00:00 | 1702.40 | 1703.02 | 1710.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 09:15:00 | 1751.90 | 1721.29 | 1717.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 1751.90 | 1721.29 | 1717.64 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 13:15:00 | 1712.30 | 1722.93 | 1723.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 1710.00 | 1720.34 | 1722.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 14:15:00 | 1700.30 | 1698.52 | 1707.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 14:15:00 | 1700.30 | 1698.52 | 1707.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 1700.30 | 1698.52 | 1707.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 1700.30 | 1698.52 | 1707.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1689.80 | 1695.57 | 1704.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 15:00:00 | 1680.20 | 1694.54 | 1701.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 15:15:00 | 1673.20 | 1672.53 | 1683.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 10:15:00 | 1722.10 | 1691.00 | 1689.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1722.10 | 1691.00 | 1689.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 10:15:00 | 1745.30 | 1722.65 | 1708.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 09:15:00 | 1790.50 | 1791.86 | 1776.02 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 13:30:00 | 1814.20 | 1797.87 | 1783.87 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 15:15:00 | 1790.00 | 1800.16 | 1787.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 09:15:00 | 1828.50 | 1800.16 | 1787.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 15:15:00 | 1823.00 | 1827.62 | 1818.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:30:00 | 1831.00 | 1829.96 | 1820.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 11:15:00 | 1820.40 | 1827.46 | 1821.31 | SL hit (close<ema400) qty=1.00 sl=1821.31 alert=retest1 |

### Cycle 76 — SELL (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 09:15:00 | 1876.00 | 1907.16 | 1908.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 11:15:00 | 1875.00 | 1896.00 | 1903.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 12:15:00 | 1899.90 | 1896.78 | 1902.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 12:15:00 | 1899.90 | 1896.78 | 1902.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 1899.90 | 1896.78 | 1902.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:45:00 | 1903.60 | 1896.78 | 1902.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1892.90 | 1890.99 | 1897.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:30:00 | 1887.60 | 1890.99 | 1897.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 1900.90 | 1891.37 | 1896.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:45:00 | 1907.70 | 1891.37 | 1896.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 1892.60 | 1891.62 | 1896.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 15:00:00 | 1871.20 | 1887.61 | 1893.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 1962.30 | 1901.33 | 1898.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 09:15:00 | 1962.30 | 1901.33 | 1898.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 10:15:00 | 1993.10 | 1972.55 | 1961.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 14:15:00 | 1977.60 | 1980.43 | 1969.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 15:00:00 | 1977.60 | 1980.43 | 1969.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 1987.90 | 1981.93 | 1971.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 1997.70 | 1977.36 | 1976.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 11:45:00 | 2006.60 | 1983.27 | 1979.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 14:15:00 | 2053.30 | 2071.08 | 2071.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 14:15:00 | 2053.30 | 2071.08 | 2071.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 15:15:00 | 2020.00 | 2060.87 | 2066.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 12:15:00 | 2072.70 | 2060.30 | 2064.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 12:15:00 | 2072.70 | 2060.30 | 2064.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 2072.70 | 2060.30 | 2064.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:45:00 | 2091.10 | 2060.30 | 2064.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 2072.10 | 2062.66 | 2065.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:15:00 | 2075.00 | 2062.66 | 2065.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 2064.00 | 2063.26 | 2064.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:15:00 | 2078.50 | 2063.26 | 2064.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 2071.10 | 2064.82 | 2065.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:00:00 | 2071.10 | 2064.82 | 2065.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 2086.60 | 2069.18 | 2067.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 12:15:00 | 2097.70 | 2075.77 | 2070.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 11:15:00 | 2086.60 | 2092.17 | 2082.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 11:15:00 | 2086.60 | 2092.17 | 2082.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 2086.60 | 2092.17 | 2082.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 12:00:00 | 2086.60 | 2092.17 | 2082.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 2076.20 | 2088.76 | 2082.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 14:00:00 | 2076.20 | 2088.76 | 2082.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 2083.70 | 2087.74 | 2082.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:15:00 | 2070.80 | 2087.74 | 2082.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 2070.80 | 2084.36 | 2081.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 09:15:00 | 2081.70 | 2084.36 | 2081.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 2079.70 | 2083.42 | 2081.50 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 10:15:00 | 2060.70 | 2078.88 | 2079.61 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 2087.20 | 2074.71 | 2074.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 11:15:00 | 2120.10 | 2086.25 | 2079.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 2090.90 | 2093.30 | 2085.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 14:15:00 | 2090.90 | 2093.30 | 2085.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 2090.90 | 2093.30 | 2085.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 2090.90 | 2093.30 | 2085.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 2069.90 | 2088.62 | 2083.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 12:15:00 | 2107.00 | 2091.84 | 2086.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 12:15:00 | 2106.00 | 2115.31 | 2105.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 14:30:00 | 2106.10 | 2110.52 | 2105.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 09:15:00 | 2074.00 | 2101.63 | 2101.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 2074.00 | 2101.63 | 2101.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 10:15:00 | 2048.80 | 2091.06 | 2097.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 12:15:00 | 2092.50 | 2089.88 | 2095.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 12:15:00 | 2092.50 | 2089.88 | 2095.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 2092.50 | 2089.88 | 2095.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:45:00 | 2096.70 | 2089.88 | 2095.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 2102.00 | 2092.31 | 2096.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 2108.00 | 2092.31 | 2096.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 2102.00 | 2094.25 | 2096.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:30:00 | 2105.00 | 2094.25 | 2096.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 09:15:00 | 2107.30 | 2098.58 | 2098.25 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 11:15:00 | 2092.30 | 2097.01 | 2097.57 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 13:15:00 | 2105.00 | 2099.24 | 2098.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 09:15:00 | 2108.00 | 2102.01 | 2100.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 2102.20 | 2103.25 | 2101.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 11:15:00 | 2102.20 | 2103.25 | 2101.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 2102.20 | 2103.25 | 2101.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:45:00 | 2101.50 | 2103.25 | 2101.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 2118.90 | 2106.38 | 2102.64 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 2086.30 | 2100.58 | 2102.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 11:15:00 | 2067.10 | 2090.62 | 2097.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 2097.30 | 2087.34 | 2092.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 2097.30 | 2087.34 | 2092.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 2097.30 | 2087.34 | 2092.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:45:00 | 2095.00 | 2087.34 | 2092.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 2088.40 | 2087.55 | 2092.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 14:30:00 | 2082.70 | 2093.56 | 2094.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 15:15:00 | 2100.00 | 2094.85 | 2094.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 15:15:00 | 2100.00 | 2094.85 | 2094.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 09:15:00 | 2119.40 | 2099.76 | 2096.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 09:15:00 | 2609.40 | 2617.29 | 2571.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 2609.40 | 2617.29 | 2571.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 2609.40 | 2617.29 | 2571.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 11:30:00 | 2641.50 | 2619.69 | 2580.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 09:45:00 | 2651.90 | 2627.51 | 2599.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 09:30:00 | 2645.00 | 2645.97 | 2625.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 13:00:00 | 2633.80 | 2636.08 | 2634.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 2628.20 | 2634.50 | 2634.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:00:00 | 2628.20 | 2634.50 | 2634.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 2637.50 | 2635.10 | 2634.35 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-24 15:15:00 | 2624.00 | 2632.88 | 2633.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 15:15:00 | 2624.00 | 2632.88 | 2633.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 2599.90 | 2625.60 | 2629.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 2650.00 | 2618.22 | 2623.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 2650.00 | 2618.22 | 2623.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 2650.00 | 2618.22 | 2623.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:00:00 | 2650.00 | 2618.22 | 2623.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 10:15:00 | 2668.00 | 2628.18 | 2627.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 15:15:00 | 2688.40 | 2669.23 | 2654.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 14:15:00 | 2666.00 | 2675.98 | 2664.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 14:15:00 | 2666.00 | 2675.98 | 2664.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 2666.00 | 2675.98 | 2664.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 15:00:00 | 2666.00 | 2675.98 | 2664.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 2669.90 | 2674.76 | 2665.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 2683.60 | 2674.76 | 2665.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 2657.50 | 2671.31 | 2664.67 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 12:15:00 | 2631.50 | 2657.96 | 2659.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 2611.30 | 2641.08 | 2650.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 12:15:00 | 2634.00 | 2632.44 | 2643.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-01 13:00:00 | 2634.00 | 2632.44 | 2643.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 2664.80 | 2628.94 | 2637.51 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 2647.20 | 2642.06 | 2641.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 15:15:00 | 2656.00 | 2644.85 | 2642.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 09:15:00 | 2635.70 | 2643.02 | 2642.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 09:15:00 | 2635.70 | 2643.02 | 2642.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 2635.70 | 2643.02 | 2642.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:00:00 | 2635.70 | 2643.02 | 2642.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 2638.80 | 2642.18 | 2642.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 11:15:00 | 2647.00 | 2642.18 | 2642.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 12:00:00 | 2643.20 | 2642.38 | 2642.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 12:15:00 | 2639.60 | 2641.83 | 2641.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 12:15:00 | 2639.60 | 2641.83 | 2641.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 2613.10 | 2634.76 | 2638.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 14:15:00 | 2632.00 | 2624.25 | 2630.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 14:15:00 | 2632.00 | 2624.25 | 2630.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 2632.00 | 2624.25 | 2630.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 15:00:00 | 2632.00 | 2624.25 | 2630.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 2624.30 | 2624.26 | 2629.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 09:30:00 | 2611.00 | 2615.21 | 2625.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 09:15:00 | 2651.30 | 2598.20 | 2601.50 | SL hit (close>static) qty=1.00 sl=2640.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 10:15:00 | 2690.50 | 2616.66 | 2609.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 2759.00 | 2682.46 | 2649.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 14:15:00 | 2765.00 | 2765.72 | 2730.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 15:00:00 | 2765.00 | 2765.72 | 2730.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 2689.30 | 2748.88 | 2728.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:45:00 | 2685.90 | 2748.88 | 2728.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 2682.20 | 2735.55 | 2724.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 2753.90 | 2720.04 | 2719.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 2765.10 | 2795.26 | 2799.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 2765.10 | 2795.26 | 2799.06 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 11:15:00 | 2801.60 | 2787.67 | 2785.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 09:15:00 | 2919.30 | 2815.24 | 2799.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 2916.50 | 2919.32 | 2882.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 14:00:00 | 2916.50 | 2919.32 | 2882.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 2895.60 | 2917.87 | 2891.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:00:00 | 2895.60 | 2917.87 | 2891.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 2902.60 | 2914.82 | 2892.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:30:00 | 2902.80 | 2914.82 | 2892.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 2894.10 | 2910.67 | 2892.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:30:00 | 2900.40 | 2910.67 | 2892.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 2896.30 | 2907.80 | 2892.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 12:45:00 | 2892.10 | 2907.80 | 2892.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 2907.00 | 2907.64 | 2894.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:45:00 | 2906.60 | 2907.64 | 2894.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 2900.40 | 2911.70 | 2899.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:30:00 | 2883.40 | 2911.70 | 2899.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 2882.00 | 2905.76 | 2898.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 2882.00 | 2905.76 | 2898.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 2914.20 | 2907.45 | 2899.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 12:30:00 | 2931.00 | 2917.94 | 2905.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:30:00 | 2954.00 | 2949.17 | 2936.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 13:15:00 | 2920.10 | 2938.78 | 2940.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 13:15:00 | 2920.10 | 2938.78 | 2940.28 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 2958.30 | 2942.13 | 2941.34 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 12:15:00 | 2913.80 | 2937.98 | 2939.86 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 14:15:00 | 2949.80 | 2937.82 | 2937.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 09:15:00 | 2992.30 | 2949.07 | 2942.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 15:15:00 | 3046.20 | 3049.34 | 3033.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 09:15:00 | 3045.40 | 3049.34 | 3033.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 3061.60 | 3057.90 | 3046.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 10:30:00 | 3074.00 | 3058.46 | 3047.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 11:30:00 | 3070.40 | 3060.77 | 3049.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 12:00:00 | 3070.00 | 3060.77 | 3049.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 15:15:00 | 3030.20 | 3051.64 | 3048.91 | SL hit (close<static) qty=1.00 sl=3035.00 alert=retest2 |

### Cycle 100 — SELL (started 2025-09-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 15:15:00 | 3033.00 | 3050.34 | 3050.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 10:15:00 | 3008.70 | 3038.07 | 3044.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 09:15:00 | 2900.50 | 2886.05 | 2916.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 2900.50 | 2886.05 | 2916.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 2900.50 | 2886.05 | 2916.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:45:00 | 2895.10 | 2886.05 | 2916.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 2896.50 | 2888.14 | 2915.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:45:00 | 2906.40 | 2888.14 | 2915.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 2841.40 | 2821.09 | 2849.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 12:15:00 | 2790.00 | 2815.79 | 2830.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 2871.00 | 2828.57 | 2834.27 | SL hit (close>static) qty=1.00 sl=2855.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 2885.00 | 2839.86 | 2838.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 10:15:00 | 2898.10 | 2866.62 | 2856.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 14:15:00 | 2920.40 | 2935.82 | 2911.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 14:15:00 | 2920.40 | 2935.82 | 2911.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 2920.40 | 2935.82 | 2911.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:30:00 | 2900.80 | 2935.82 | 2911.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 2910.00 | 2930.66 | 2911.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 2966.30 | 2930.66 | 2911.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 10:15:00 | 2937.20 | 2945.30 | 2946.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 2937.20 | 2945.30 | 2946.03 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 11:15:00 | 2956.30 | 2947.50 | 2946.96 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 2922.80 | 2944.73 | 2945.93 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 09:15:00 | 3095.10 | 2974.33 | 2958.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 10:15:00 | 3233.00 | 3026.07 | 2983.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 09:15:00 | 3124.80 | 3130.65 | 3068.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-15 09:45:00 | 3125.00 | 3130.65 | 3068.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 3104.40 | 3117.87 | 3090.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:30:00 | 3104.30 | 3117.87 | 3090.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 3074.50 | 3105.70 | 3091.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 13:00:00 | 3074.50 | 3105.70 | 3091.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 3078.60 | 3100.28 | 3090.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 13:30:00 | 3075.80 | 3100.28 | 3090.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 3156.90 | 3112.25 | 3098.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:30:00 | 3218.60 | 3142.52 | 3124.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 11:15:00 | 3100.30 | 3144.13 | 3144.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 11:15:00 | 3100.30 | 3144.13 | 3144.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 15:15:00 | 3082.00 | 3115.89 | 3129.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 14:15:00 | 3121.10 | 3092.20 | 3109.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 14:15:00 | 3121.10 | 3092.20 | 3109.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 3121.10 | 3092.20 | 3109.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 15:00:00 | 3121.10 | 3092.20 | 3109.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 3100.00 | 3093.76 | 3108.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:15:00 | 3191.70 | 3093.76 | 3108.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 3200.30 | 3115.07 | 3116.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:30:00 | 3225.30 | 3115.07 | 3116.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 10:15:00 | 3202.40 | 3132.54 | 3124.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 13:15:00 | 3221.20 | 3170.67 | 3145.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 3182.00 | 3187.00 | 3160.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 10:00:00 | 3182.00 | 3187.00 | 3160.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 108 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 3066.00 | 3159.96 | 3160.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 14:15:00 | 3056.20 | 3098.41 | 3126.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 09:15:00 | 3140.60 | 3105.83 | 3124.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 3140.60 | 3105.83 | 3124.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 3140.60 | 3105.83 | 3124.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:30:00 | 3148.40 | 3105.83 | 3124.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 3150.40 | 3114.74 | 3126.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:00:00 | 3150.40 | 3114.74 | 3126.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 3121.60 | 3116.11 | 3126.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:30:00 | 3146.10 | 3116.11 | 3126.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 3114.40 | 3115.77 | 3125.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 10:00:00 | 3102.10 | 3116.85 | 3123.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 3135.50 | 3106.61 | 3112.74 | SL hit (close>static) qty=1.00 sl=3126.20 alert=retest2 |

### Cycle 109 — BUY (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 12:15:00 | 3135.00 | 3113.19 | 3111.68 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 13:15:00 | 3100.30 | 3110.62 | 3110.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 10:15:00 | 3091.20 | 3104.42 | 3107.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 3114.00 | 3106.34 | 3107.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 11:15:00 | 3114.00 | 3106.34 | 3107.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 3114.00 | 3106.34 | 3107.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:00:00 | 3114.00 | 3106.34 | 3107.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 3117.00 | 3108.47 | 3108.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 3117.00 | 3108.47 | 3108.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 13:15:00 | 3129.40 | 3112.66 | 3110.51 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 11:15:00 | 3100.00 | 3110.64 | 3110.71 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 14:15:00 | 3120.90 | 3111.23 | 3110.78 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 3091.00 | 3108.62 | 3109.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 10:15:00 | 3071.30 | 3101.15 | 3106.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 12:15:00 | 3074.70 | 3070.39 | 3082.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-12 13:00:00 | 3074.70 | 3070.39 | 3082.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 3078.80 | 3072.07 | 3082.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:00:00 | 3078.80 | 3072.07 | 3082.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 3097.60 | 3077.18 | 3083.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 15:00:00 | 3097.60 | 3077.18 | 3083.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 3081.00 | 3077.94 | 3083.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 3074.30 | 3077.94 | 3083.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 09:15:00 | 2920.59 | 2974.41 | 3004.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-21 11:15:00 | 2888.90 | 2881.39 | 2914.26 | SL hit (close>ema200) qty=0.50 sl=2881.39 alert=retest2 |

### Cycle 115 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 2900.40 | 2883.80 | 2882.91 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 2878.20 | 2884.38 | 2884.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 13:15:00 | 2873.10 | 2882.12 | 2883.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 14:15:00 | 2895.30 | 2884.76 | 2884.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 14:15:00 | 2895.30 | 2884.76 | 2884.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 2895.30 | 2884.76 | 2884.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 15:00:00 | 2895.30 | 2884.76 | 2884.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 2880.00 | 2883.81 | 2884.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 09:45:00 | 2862.30 | 2881.02 | 2883.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 10:15:00 | 2861.30 | 2881.02 | 2883.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 09:15:00 | 2929.00 | 2889.57 | 2885.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 2929.00 | 2889.57 | 2885.08 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 10:15:00 | 2900.80 | 2907.76 | 2907.86 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 14:15:00 | 2933.70 | 2911.84 | 2909.47 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 11:15:00 | 2893.10 | 2907.91 | 2908.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 12:15:00 | 2889.00 | 2904.13 | 2906.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 15:15:00 | 2910.00 | 2902.39 | 2905.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 15:15:00 | 2910.00 | 2902.39 | 2905.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 2910.00 | 2902.39 | 2905.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:30:00 | 2876.10 | 2895.37 | 2901.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:00:00 | 2883.80 | 2858.16 | 2865.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 13:45:00 | 2880.50 | 2868.12 | 2868.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 14:15:00 | 2876.70 | 2869.83 | 2869.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 14:15:00 | 2876.70 | 2869.83 | 2869.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 10:15:00 | 2892.60 | 2875.78 | 2872.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 15:15:00 | 2880.80 | 2882.47 | 2877.44 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 09:30:00 | 2909.10 | 2888.37 | 2880.58 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 12:15:00 | 2898.60 | 2892.93 | 2884.24 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 2989.00 | 2988.91 | 2980.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 2989.00 | 2988.91 | 2980.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 3009.60 | 2993.70 | 2984.52 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-19 12:15:00 | 2980.40 | 2991.51 | 2985.89 | SL hit (close<ema400) qty=1.00 sl=2985.89 alert=retest1 |

### Cycle 122 — SELL (started 2026-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 13:15:00 | 3080.50 | 3086.49 | 3086.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 15:15:00 | 3079.00 | 3083.95 | 3085.65 | Break + close below crossover candle low |

### Cycle 123 — BUY (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 09:15:00 | 3105.10 | 3088.18 | 3087.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 10:15:00 | 3134.80 | 3097.50 | 3091.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 15:15:00 | 3116.00 | 3116.42 | 3104.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 09:15:00 | 3141.30 | 3116.42 | 3104.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 3154.60 | 3124.06 | 3109.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 10:45:00 | 3165.90 | 3150.67 | 3133.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 11:15:00 | 3170.00 | 3150.67 | 3133.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 12:15:00 | 3167.70 | 3152.14 | 3135.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 09:30:00 | 3181.00 | 3162.39 | 3147.70 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 3152.70 | 3159.68 | 3150.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:00:00 | 3152.70 | 3159.68 | 3150.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 3149.40 | 3157.63 | 3150.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:30:00 | 3140.00 | 3157.63 | 3150.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 3142.50 | 3154.60 | 3149.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 15:00:00 | 3142.50 | 3154.60 | 3149.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 3140.00 | 3151.68 | 3148.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:15:00 | 3115.00 | 3151.68 | 3148.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-09 10:15:00 | 3127.00 | 3142.97 | 3144.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2026-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 10:15:00 | 3127.00 | 3142.97 | 3144.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 11:15:00 | 3101.30 | 3124.39 | 3134.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 3130.00 | 3121.20 | 3129.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 15:15:00 | 3130.00 | 3121.20 | 3129.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 3130.00 | 3121.20 | 3129.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 3162.60 | 3121.20 | 3129.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 3167.70 | 3130.50 | 3132.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:30:00 | 3163.20 | 3130.50 | 3132.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 10:15:00 | 3175.70 | 3139.54 | 3136.56 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 09:15:00 | 3098.10 | 3132.11 | 3134.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 10:15:00 | 3078.50 | 3121.39 | 3129.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 09:15:00 | 3115.80 | 3101.18 | 3113.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 3115.80 | 3101.18 | 3113.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 3115.80 | 3101.18 | 3113.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:45:00 | 3113.30 | 3101.18 | 3113.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 3121.10 | 3105.16 | 3114.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:00:00 | 3121.10 | 3105.16 | 3114.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 3125.00 | 3109.13 | 3115.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:45:00 | 3124.40 | 3109.13 | 3115.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 3094.00 | 3101.42 | 3109.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 3067.10 | 3101.42 | 3109.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 13:15:00 | 3081.00 | 3103.29 | 3108.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 15:15:00 | 3078.80 | 3099.00 | 3105.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 2913.74 | 2960.29 | 2995.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 2926.95 | 2960.29 | 2995.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 2924.86 | 2960.29 | 2995.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-27 11:15:00 | 2760.39 | 2895.61 | 2958.60 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 127 — BUY (started 2026-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 15:15:00 | 2910.00 | 2901.19 | 2900.59 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 09:15:00 | 2870.10 | 2894.97 | 2897.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 13:15:00 | 2838.10 | 2873.30 | 2885.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 14:15:00 | 2882.00 | 2875.04 | 2885.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 14:15:00 | 2882.00 | 2875.04 | 2885.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 2882.00 | 2875.04 | 2885.46 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 11:15:00 | 2902.30 | 2890.70 | 2890.21 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 2847.80 | 2889.39 | 2891.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 2830.00 | 2877.51 | 2885.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 2878.20 | 2861.01 | 2873.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 2878.20 | 2861.01 | 2873.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 2878.20 | 2861.01 | 2873.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 2878.20 | 2861.01 | 2873.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 2877.00 | 2864.21 | 2873.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 2900.90 | 2864.21 | 2873.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2930.00 | 2877.37 | 2878.98 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 2926.00 | 2887.09 | 2883.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 2951.30 | 2899.94 | 2889.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 15:15:00 | 2942.00 | 2951.70 | 2933.40 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 09:15:00 | 2962.50 | 2951.70 | 2933.40 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 10:15:00 | 2965.50 | 2951.44 | 2934.94 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 2939.80 | 2949.11 | 2935.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 2939.80 | 2949.11 | 2935.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 2934.50 | 2946.19 | 2935.30 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-05 11:15:00 | 2934.50 | 2946.19 | 2935.30 | SL hit (close<ema400) qty=1.00 sl=2935.30 alert=retest1 |

### Cycle 132 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 3127.40 | 3170.65 | 3171.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 3100.90 | 3127.61 | 3138.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 09:15:00 | 3070.30 | 3051.56 | 3074.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 09:15:00 | 3070.30 | 3051.56 | 3074.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 3070.30 | 3051.56 | 3074.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:45:00 | 3073.90 | 3051.56 | 3074.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 3048.50 | 3050.95 | 3072.46 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 3088.90 | 3069.54 | 3067.34 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 13:15:00 | 3058.20 | 3065.03 | 3065.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 3028.40 | 3057.82 | 3062.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 14:15:00 | 3009.50 | 2995.97 | 3017.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 14:15:00 | 3009.50 | 2995.97 | 3017.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 3009.50 | 2995.97 | 3017.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 3009.50 | 2995.97 | 3017.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 3021.90 | 3001.16 | 3017.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 2985.00 | 3001.16 | 3017.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 10:15:00 | 3009.50 | 2979.88 | 2977.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 3009.50 | 2979.88 | 2977.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-27 14:15:00 | 3032.40 | 2996.13 | 2989.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 13:15:00 | 3014.10 | 3018.59 | 3005.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-30 13:45:00 | 3015.50 | 3018.59 | 3005.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 3000.40 | 3014.95 | 3004.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 15:00:00 | 3000.40 | 3014.95 | 3004.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 3010.50 | 3014.06 | 3005.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 3101.20 | 3014.06 | 3005.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-09 09:15:00 | 3411.32 | 3385.30 | 3321.41 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2026-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 15:15:00 | 3650.00 | 3653.24 | 3653.30 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 09:15:00 | 3656.50 | 3653.89 | 3653.59 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 3642.40 | 3651.59 | 3652.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 13:15:00 | 3632.00 | 3646.62 | 3650.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 09:15:00 | 3638.90 | 3635.83 | 3643.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 09:15:00 | 3638.90 | 3635.83 | 3643.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 3638.90 | 3635.83 | 3643.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 13:30:00 | 3599.10 | 3625.84 | 3635.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 09:30:00 | 3596.50 | 3616.81 | 3628.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 10:00:00 | 3606.80 | 3573.98 | 3586.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:00:00 | 3606.50 | 3580.48 | 3588.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 3591.40 | 3582.92 | 3587.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:00:00 | 3591.40 | 3582.92 | 3587.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 3570.00 | 3580.34 | 3586.29 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-29 14:15:00 | 3589.60 | 3587.51 | 3587.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 14:15:00 | 3589.60 | 3587.51 | 3587.50 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 3575.00 | 3585.01 | 3586.36 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 09:15:00 | 3616.30 | 3591.26 | 3589.08 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 11:15:00 | 3605.00 | 3616.54 | 3616.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 15:15:00 | 3602.30 | 3611.38 | 3613.99 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-16 10:00:00 | 1975.50 | 2024-05-18 12:15:00 | 1972.50 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2024-05-16 10:30:00 | 1975.25 | 2024-05-18 12:15:00 | 1972.50 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2024-05-16 12:15:00 | 1975.25 | 2024-05-18 12:15:00 | 1972.50 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2024-05-16 14:00:00 | 1979.98 | 2024-05-18 12:15:00 | 1972.50 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2024-05-28 12:30:00 | 2058.35 | 2024-05-31 12:15:00 | 2052.48 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2024-05-31 10:30:00 | 2055.18 | 2024-05-31 12:15:00 | 2052.48 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2024-06-07 10:00:00 | 1980.00 | 2024-06-12 11:15:00 | 1963.50 | STOP_HIT | 1.00 | 0.83% |
| SELL | retest2 | 2024-06-27 10:15:00 | 1922.50 | 2024-06-28 09:15:00 | 1968.23 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2024-06-27 11:45:00 | 1924.70 | 2024-06-28 09:15:00 | 1968.23 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2024-06-27 12:15:00 | 1922.50 | 2024-06-28 09:15:00 | 1968.23 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2024-06-27 12:45:00 | 1922.50 | 2024-06-28 09:15:00 | 1968.23 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2024-07-01 15:15:00 | 1962.53 | 2024-07-02 10:15:00 | 1943.63 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-07-04 10:30:00 | 1932.50 | 2024-07-04 15:15:00 | 1950.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-07-04 11:30:00 | 1934.00 | 2024-07-04 15:15:00 | 1950.00 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-07-23 09:15:00 | 1897.00 | 2024-07-30 11:15:00 | 1880.53 | STOP_HIT | 1.00 | 0.87% |
| SELL | retest2 | 2024-07-23 12:15:00 | 1885.00 | 2024-07-30 11:15:00 | 1880.53 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2024-07-23 13:00:00 | 1890.38 | 2024-07-30 11:15:00 | 1880.53 | STOP_HIT | 1.00 | 0.52% |
| SELL | retest2 | 2024-07-23 14:15:00 | 1887.65 | 2024-07-30 11:15:00 | 1880.53 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2024-07-24 15:00:00 | 1900.05 | 2024-07-30 11:15:00 | 1880.53 | STOP_HIT | 1.00 | 1.03% |
| SELL | retest2 | 2024-08-08 15:00:00 | 1792.15 | 2024-08-09 09:15:00 | 1823.88 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-08-30 12:15:00 | 1891.48 | 2024-09-09 10:15:00 | 1910.85 | STOP_HIT | 1.00 | 1.02% |
| BUY | retest2 | 2024-09-13 09:45:00 | 1955.83 | 2024-09-18 12:15:00 | 1957.93 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2024-09-13 11:00:00 | 1952.85 | 2024-09-18 12:15:00 | 1957.93 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2024-09-20 11:30:00 | 1898.35 | 2024-09-20 14:15:00 | 1955.40 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2024-09-26 09:15:00 | 1967.30 | 2024-09-27 14:15:00 | 1930.90 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2024-09-26 14:30:00 | 1959.48 | 2024-09-27 14:15:00 | 1930.90 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-09-27 09:15:00 | 1955.58 | 2024-09-27 14:15:00 | 1930.90 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-09-27 12:30:00 | 1957.08 | 2024-09-27 14:15:00 | 1930.90 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-10-04 09:15:00 | 1921.25 | 2024-10-04 10:15:00 | 1952.80 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2024-10-11 09:15:00 | 2072.90 | 2024-10-17 14:15:00 | 2036.55 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2024-10-14 09:15:00 | 2054.60 | 2024-10-17 14:15:00 | 2036.55 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-10-15 09:30:00 | 2055.90 | 2024-10-17 14:15:00 | 2036.55 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-10-15 10:15:00 | 2054.53 | 2024-10-17 14:15:00 | 2036.55 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-10-15 13:15:00 | 2050.00 | 2024-10-17 14:15:00 | 2036.55 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-10-15 13:45:00 | 2064.32 | 2024-10-17 14:15:00 | 2036.55 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-10-16 09:30:00 | 2059.07 | 2024-10-17 14:15:00 | 2036.55 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-10-16 10:00:00 | 2059.35 | 2024-10-17 14:15:00 | 2036.55 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-10-24 11:30:00 | 2041.75 | 2024-10-30 15:15:00 | 1939.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 15:15:00 | 2043.98 | 2024-10-30 15:15:00 | 1941.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 11:30:00 | 2041.75 | 2024-10-31 09:15:00 | 1991.60 | STOP_HIT | 0.50 | 2.46% |
| SELL | retest2 | 2024-10-24 15:15:00 | 2043.98 | 2024-10-31 09:15:00 | 1991.60 | STOP_HIT | 0.50 | 2.56% |
| BUY | retest2 | 2024-11-04 13:45:00 | 2020.35 | 2024-11-06 11:15:00 | 2002.53 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-11-04 14:30:00 | 2019.60 | 2024-11-06 11:15:00 | 2002.53 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-11-05 09:30:00 | 2019.00 | 2024-11-06 11:15:00 | 2002.53 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-11-05 11:00:00 | 2022.18 | 2024-11-06 11:15:00 | 2002.53 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-11-12 12:15:00 | 1959.73 | 2024-11-18 11:15:00 | 1971.75 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2024-11-12 14:45:00 | 1955.33 | 2024-11-18 11:15:00 | 1971.75 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-11-13 09:15:00 | 1920.00 | 2024-11-18 11:15:00 | 1971.75 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2024-11-28 14:45:00 | 2062.07 | 2024-12-09 10:15:00 | 2268.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-28 15:15:00 | 2059.00 | 2024-12-09 10:15:00 | 2264.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-29 10:15:00 | 2058.48 | 2024-12-09 10:15:00 | 2264.33 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-30 09:15:00 | 1989.73 | 2025-01-03 14:15:00 | 1980.00 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2024-12-30 10:00:00 | 1988.95 | 2025-01-03 14:15:00 | 1980.00 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2024-12-30 12:15:00 | 1985.03 | 2025-01-03 14:15:00 | 1980.00 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2024-12-30 13:15:00 | 1990.00 | 2025-01-03 14:15:00 | 1980.00 | STOP_HIT | 1.00 | 0.50% |
| SELL | retest2 | 2025-01-01 12:15:00 | 1968.38 | 2025-01-03 14:15:00 | 1980.00 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-01-01 12:45:00 | 1964.83 | 2025-01-03 14:15:00 | 1980.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-01-01 13:45:00 | 1967.50 | 2025-01-03 14:15:00 | 1980.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-01-01 14:30:00 | 1969.23 | 2025-01-03 14:15:00 | 1980.00 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-01-08 10:15:00 | 1927.75 | 2025-01-09 09:15:00 | 1970.50 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2025-01-29 09:15:00 | 1747.50 | 2025-01-30 09:15:00 | 1783.45 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-01-29 11:15:00 | 1751.23 | 2025-01-30 09:15:00 | 1783.45 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-02-11 09:30:00 | 1810.08 | 2025-02-18 11:15:00 | 1829.40 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-02-28 12:45:00 | 2004.70 | 2025-03-05 09:15:00 | 1933.95 | STOP_HIT | 1.00 | -3.53% |
| BUY | retest2 | 2025-02-28 13:15:00 | 2001.83 | 2025-03-05 09:15:00 | 1933.95 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2025-03-03 09:45:00 | 2001.53 | 2025-03-05 09:15:00 | 1933.95 | STOP_HIT | 1.00 | -3.38% |
| SELL | retest2 | 2025-03-10 09:15:00 | 1832.60 | 2025-03-11 09:15:00 | 1740.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-10 09:15:00 | 1832.60 | 2025-03-11 10:15:00 | 1802.90 | STOP_HIT | 0.50 | 1.62% |
| BUY | retest2 | 2025-03-25 14:15:00 | 1814.15 | 2025-04-01 15:15:00 | 1825.10 | STOP_HIT | 1.00 | 0.60% |
| BUY | retest2 | 2025-04-23 10:15:00 | 1779.00 | 2025-04-25 09:15:00 | 1730.00 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-04-23 11:00:00 | 1781.90 | 2025-04-25 09:15:00 | 1730.00 | STOP_HIT | 1.00 | -2.91% |
| BUY | retest2 | 2025-04-24 09:30:00 | 1790.00 | 2025-04-25 09:15:00 | 1730.00 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2025-04-30 15:15:00 | 1692.00 | 2025-05-05 09:15:00 | 1751.90 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2025-05-02 12:00:00 | 1702.40 | 2025-05-05 09:15:00 | 1751.90 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2025-05-08 15:00:00 | 1680.20 | 2025-05-12 10:15:00 | 1722.10 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2025-05-09 15:15:00 | 1673.20 | 2025-05-12 10:15:00 | 1722.10 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest1 | 2025-05-16 13:30:00 | 1814.20 | 2025-05-21 11:15:00 | 1820.40 | STOP_HIT | 1.00 | 0.34% |
| BUY | retest2 | 2025-05-19 09:15:00 | 1828.50 | 2025-05-29 09:15:00 | 1876.00 | STOP_HIT | 1.00 | 2.60% |
| BUY | retest2 | 2025-05-20 15:15:00 | 1823.00 | 2025-05-29 09:15:00 | 1876.00 | STOP_HIT | 1.00 | 2.91% |
| BUY | retest2 | 2025-05-21 09:30:00 | 1831.00 | 2025-05-29 09:15:00 | 1876.00 | STOP_HIT | 1.00 | 2.46% |
| BUY | retest2 | 2025-05-21 12:15:00 | 1825.90 | 2025-05-29 09:15:00 | 1876.00 | STOP_HIT | 1.00 | 2.74% |
| BUY | retest2 | 2025-05-21 14:45:00 | 1836.90 | 2025-05-29 09:15:00 | 1876.00 | STOP_HIT | 1.00 | 2.13% |
| BUY | retest2 | 2025-05-22 09:30:00 | 1840.40 | 2025-05-29 09:15:00 | 1876.00 | STOP_HIT | 1.00 | 1.93% |
| BUY | retest2 | 2025-05-23 09:15:00 | 1842.10 | 2025-05-29 09:15:00 | 1876.00 | STOP_HIT | 1.00 | 1.84% |
| SELL | retest2 | 2025-05-30 15:00:00 | 1871.20 | 2025-06-02 09:15:00 | 1962.30 | STOP_HIT | 1.00 | -4.87% |
| BUY | retest2 | 2025-06-11 09:15:00 | 1997.70 | 2025-06-19 14:15:00 | 2053.30 | STOP_HIT | 1.00 | 2.78% |
| BUY | retest2 | 2025-06-11 11:45:00 | 2006.60 | 2025-06-19 14:15:00 | 2053.30 | STOP_HIT | 1.00 | 2.33% |
| BUY | retest2 | 2025-06-30 12:15:00 | 2107.00 | 2025-07-02 09:15:00 | 2074.00 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-07-01 12:15:00 | 2106.00 | 2025-07-02 09:15:00 | 2074.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-07-01 14:30:00 | 2106.10 | 2025-07-02 09:15:00 | 2074.00 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-07-09 14:30:00 | 2082.70 | 2025-07-09 15:15:00 | 2100.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-07-21 11:30:00 | 2641.50 | 2025-07-24 15:15:00 | 2624.00 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-07-22 09:45:00 | 2651.90 | 2025-07-24 15:15:00 | 2624.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-07-23 09:30:00 | 2645.00 | 2025-07-24 15:15:00 | 2624.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-07-24 13:00:00 | 2633.80 | 2025-07-24 15:15:00 | 2624.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-08-05 11:15:00 | 2647.00 | 2025-08-05 12:15:00 | 2639.60 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-08-05 12:00:00 | 2643.20 | 2025-08-05 12:15:00 | 2639.60 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2025-08-07 09:30:00 | 2611.00 | 2025-08-11 09:15:00 | 2651.30 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-08-18 09:15:00 | 2753.90 | 2025-08-26 09:15:00 | 2765.10 | STOP_HIT | 1.00 | 0.41% |
| BUY | retest2 | 2025-09-04 12:30:00 | 2931.00 | 2025-09-09 13:15:00 | 2920.10 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-09-08 09:30:00 | 2954.00 | 2025-09-09 13:15:00 | 2920.10 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-09-19 10:30:00 | 3074.00 | 2025-09-19 15:15:00 | 3030.20 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-09-19 11:30:00 | 3070.40 | 2025-09-19 15:15:00 | 3030.20 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-09-19 12:00:00 | 3070.00 | 2025-09-19 15:15:00 | 3030.20 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-09-22 09:15:00 | 3070.00 | 2025-09-22 15:15:00 | 3033.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-10-01 12:15:00 | 2790.00 | 2025-10-01 13:15:00 | 2871.00 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2025-10-08 09:15:00 | 2966.30 | 2025-10-13 10:15:00 | 2937.20 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-10-20 09:30:00 | 3218.60 | 2025-10-23 11:15:00 | 3100.30 | STOP_HIT | 1.00 | -3.68% |
| SELL | retest2 | 2025-10-31 10:00:00 | 3102.10 | 2025-11-03 09:15:00 | 3135.50 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-11-03 11:30:00 | 3105.30 | 2025-11-04 11:15:00 | 3134.10 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-11-03 13:15:00 | 3106.10 | 2025-11-04 11:15:00 | 3134.10 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-11-13 09:15:00 | 3074.30 | 2025-11-19 09:15:00 | 2920.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-13 09:15:00 | 3074.30 | 2025-11-21 11:15:00 | 2888.90 | STOP_HIT | 0.50 | 6.03% |
| SELL | retest2 | 2025-11-28 09:45:00 | 2862.30 | 2025-12-01 09:15:00 | 2929.00 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-11-28 10:15:00 | 2861.30 | 2025-12-01 09:15:00 | 2929.00 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-12-08 09:30:00 | 2876.10 | 2025-12-10 14:15:00 | 2876.70 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-12-10 10:00:00 | 2883.80 | 2025-12-10 14:15:00 | 2876.70 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2025-12-10 13:45:00 | 2880.50 | 2025-12-10 14:15:00 | 2876.70 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest1 | 2025-12-12 09:30:00 | 2909.10 | 2025-12-19 12:15:00 | 2980.40 | STOP_HIT | 1.00 | 2.45% |
| BUY | retest1 | 2025-12-12 12:15:00 | 2898.60 | 2025-12-19 12:15:00 | 2980.40 | STOP_HIT | 1.00 | 2.82% |
| BUY | retest2 | 2025-12-23 14:00:00 | 3026.50 | 2026-01-02 13:15:00 | 3080.50 | STOP_HIT | 1.00 | 1.78% |
| BUY | retest2 | 2025-12-23 14:45:00 | 3039.30 | 2026-01-02 13:15:00 | 3080.50 | STOP_HIT | 1.00 | 1.36% |
| BUY | retest2 | 2025-12-24 09:15:00 | 3056.10 | 2026-01-02 13:15:00 | 3080.50 | STOP_HIT | 1.00 | 0.80% |
| BUY | retest2 | 2025-12-26 15:00:00 | 3049.50 | 2026-01-02 13:15:00 | 3080.50 | STOP_HIT | 1.00 | 1.02% |
| BUY | retest2 | 2025-12-29 10:45:00 | 3113.80 | 2026-01-02 13:15:00 | 3080.50 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-12-31 09:45:00 | 3098.80 | 2026-01-02 13:15:00 | 3080.50 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-12-31 12:00:00 | 3096.30 | 2026-01-02 13:15:00 | 3080.50 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-12-31 12:30:00 | 3101.80 | 2026-01-02 13:15:00 | 3080.50 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2026-01-07 10:45:00 | 3165.90 | 2026-01-09 10:15:00 | 3127.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2026-01-07 11:15:00 | 3170.00 | 2026-01-09 10:15:00 | 3127.00 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2026-01-07 12:15:00 | 3167.70 | 2026-01-09 10:15:00 | 3127.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2026-01-08 09:30:00 | 3181.00 | 2026-01-09 10:15:00 | 3127.00 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-01-19 09:15:00 | 3067.10 | 2026-01-27 09:15:00 | 2913.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 13:15:00 | 3081.00 | 2026-01-27 09:15:00 | 2926.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 15:15:00 | 3078.80 | 2026-01-27 09:15:00 | 2924.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 09:15:00 | 3067.10 | 2026-01-27 11:15:00 | 2760.39 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-19 13:15:00 | 3081.00 | 2026-01-27 11:15:00 | 2772.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-19 15:15:00 | 3078.80 | 2026-01-27 11:15:00 | 2770.92 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2026-02-05 09:15:00 | 2962.50 | 2026-02-05 11:15:00 | 2934.50 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest1 | 2026-02-05 10:15:00 | 2965.50 | 2026-02-05 11:15:00 | 2934.50 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-02-06 11:45:00 | 2959.90 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 5.66% |
| BUY | retest2 | 2026-02-09 09:15:00 | 2961.00 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 5.62% |
| BUY | retest2 | 2026-02-09 14:15:00 | 2977.00 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 5.05% |
| BUY | retest2 | 2026-02-10 11:15:00 | 2964.30 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 5.50% |
| BUY | retest2 | 2026-02-11 11:15:00 | 2974.10 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 5.15% |
| BUY | retest2 | 2026-02-12 14:45:00 | 2989.90 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 4.60% |
| BUY | retest2 | 2026-02-13 10:45:00 | 2974.10 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 5.15% |
| BUY | retest2 | 2026-02-13 12:00:00 | 2976.70 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 5.06% |
| BUY | retest2 | 2026-02-13 14:00:00 | 3006.00 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 4.04% |
| BUY | retest2 | 2026-02-16 09:45:00 | 3003.00 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 4.14% |
| BUY | retest2 | 2026-02-16 12:15:00 | 3008.90 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 3.94% |
| BUY | retest2 | 2026-02-16 15:15:00 | 2998.90 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 4.28% |
| BUY | retest2 | 2026-02-19 09:30:00 | 3027.30 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 3.31% |
| BUY | retest2 | 2026-02-19 11:30:00 | 3034.00 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 3.08% |
| BUY | retest2 | 2026-02-20 09:30:00 | 3021.70 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 3.50% |
| BUY | retest2 | 2026-02-20 10:30:00 | 3025.10 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 3.38% |
| BUY | retest2 | 2026-02-23 09:15:00 | 3039.00 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 2.91% |
| BUY | retest2 | 2026-02-23 11:15:00 | 3025.90 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 3.35% |
| BUY | retest2 | 2026-02-24 12:15:00 | 3025.80 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 3.36% |
| BUY | retest2 | 2026-02-24 14:00:00 | 3034.00 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 3.08% |
| BUY | retest2 | 2026-02-25 10:45:00 | 3064.40 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 2.06% |
| BUY | retest2 | 2026-02-25 11:45:00 | 3061.60 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 2.15% |
| BUY | retest2 | 2026-02-25 13:00:00 | 3063.00 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 2.10% |
| BUY | retest2 | 2026-02-25 14:15:00 | 3060.00 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 2.20% |
| BUY | retest2 | 2026-02-26 11:00:00 | 3074.10 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 1.73% |
| BUY | retest2 | 2026-02-26 11:45:00 | 3073.10 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 1.77% |
| BUY | retest2 | 2026-02-26 12:30:00 | 3074.10 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 1.73% |
| BUY | retest2 | 2026-02-26 14:15:00 | 3078.80 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 1.58% |
| BUY | retest2 | 2026-03-02 10:45:00 | 3082.00 | 2026-03-09 11:15:00 | 3127.40 | STOP_HIT | 1.00 | 1.47% |
| SELL | retest2 | 2026-03-23 09:15:00 | 2985.00 | 2026-03-25 10:15:00 | 3009.50 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2026-04-01 09:15:00 | 3101.20 | 2026-04-09 09:15:00 | 3411.32 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-23 13:30:00 | 3599.10 | 2026-04-29 14:15:00 | 3589.60 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2026-04-24 09:30:00 | 3596.50 | 2026-04-29 14:15:00 | 3589.60 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2026-04-28 10:00:00 | 3606.80 | 2026-04-29 14:15:00 | 3589.60 | STOP_HIT | 1.00 | 0.48% |
| SELL | retest2 | 2026-04-28 11:00:00 | 3606.50 | 2026-04-29 14:15:00 | 3589.60 | STOP_HIT | 1.00 | 0.47% |
