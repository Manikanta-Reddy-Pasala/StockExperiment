# SBILIFE (SBILIFE)

## Backtest Summary

- **Window:** 2026-01-20 09:15:00 → 2026-05-08 15:15:00 (511 bars)
- **Last close:** 1871.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 19 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT2_SKIP | 4 |
| ALERT3 | 32 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 18 |
| PARTIAL | 5 |
| TARGET_HIT | 0 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 12
- **Target hits / Stop hits / Partials:** 0 / 18 / 5
- **Avg / median % per leg:** 0.71% / -0.60%
- **Sum % (uncompounded):** 16.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 1 | 33.3% | 0 | 3 | 0 | -1.93% | -5.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 1 | 33.3% | 0 | 3 | 0 | -1.93% | -5.8% |
| SELL (all) | 20 | 10 | 50.0% | 0 | 15 | 5 | 1.11% | 22.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 20 | 10 | 50.0% | 0 | 15 | 5 | 1.11% | 22.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 23 | 11 | 47.8% | 0 | 18 | 5 | 0.71% | 16.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 2039.10 | 2029.92 | 2029.11 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 1999.90 | 2034.48 | 2035.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 11:15:00 | 1981.10 | 2023.81 | 2030.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 11:15:00 | 2000.30 | 1995.85 | 2009.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-30 12:00:00 | 2000.30 | 1995.85 | 2009.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 1994.40 | 1995.56 | 2008.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:45:00 | 2005.00 | 1995.56 | 2008.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 2008.10 | 1998.07 | 2008.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:00:00 | 2008.10 | 1998.07 | 2008.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 1995.20 | 1997.49 | 2007.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:45:00 | 2012.90 | 1997.49 | 2007.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 2009.90 | 1999.98 | 2007.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 2003.90 | 1999.98 | 2007.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 1990.10 | 1998.00 | 2005.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 11:30:00 | 1980.70 | 1990.60 | 2000.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:00:00 | 1963.00 | 1990.60 | 2000.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 14:45:00 | 1976.20 | 1984.45 | 1995.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 10:45:00 | 1980.50 | 1979.47 | 1989.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 1992.20 | 1982.02 | 1990.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:00:00 | 1992.20 | 1982.02 | 1990.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 1996.30 | 1984.87 | 1990.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:45:00 | 1997.70 | 1984.87 | 1990.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 1991.50 | 1986.20 | 1990.81 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 2051.70 | 2003.57 | 1997.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 2051.70 | 2003.57 | 1997.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 2051.70 | 2003.57 | 1997.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 2051.70 | 2003.57 | 1997.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 2051.70 | 2003.57 | 1997.94 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 2007.90 | 2017.60 | 2018.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 10:15:00 | 1985.70 | 2011.22 | 2015.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 1997.50 | 1996.81 | 2005.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 15:00:00 | 1997.50 | 1996.81 | 2005.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 2028.40 | 2002.62 | 2006.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 2028.40 | 2002.62 | 2006.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 2033.40 | 2008.77 | 2009.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:30:00 | 2034.20 | 2008.77 | 2009.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 2025.40 | 2012.10 | 2010.76 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 11:15:00 | 2004.90 | 2011.80 | 2012.24 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2026-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 13:15:00 | 2020.90 | 2013.67 | 2013.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 15:15:00 | 2022.10 | 2016.05 | 2014.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 09:15:00 | 2005.00 | 2019.50 | 2017.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 09:15:00 | 2005.00 | 2019.50 | 2017.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 2005.00 | 2019.50 | 2017.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:45:00 | 2002.70 | 2019.50 | 2017.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 2010.10 | 2017.62 | 2017.24 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 2012.20 | 2016.53 | 2016.78 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 14:15:00 | 2020.40 | 2017.20 | 2016.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 09:15:00 | 2031.30 | 2020.47 | 2018.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 11:15:00 | 2017.00 | 2019.81 | 2018.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 11:15:00 | 2017.00 | 2019.81 | 2018.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 2017.00 | 2019.81 | 2018.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:00:00 | 2017.00 | 2019.81 | 2018.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 2022.50 | 2020.35 | 2018.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 14:00:00 | 2031.90 | 2022.66 | 2020.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 14:15:00 | 2072.10 | 2084.61 | 2085.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2026-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 14:15:00 | 2072.10 | 2084.61 | 2085.85 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 2091.70 | 2086.29 | 2086.23 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 11:15:00 | 2077.00 | 2084.43 | 2085.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 12:15:00 | 2074.20 | 2082.38 | 2084.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 14:15:00 | 2081.60 | 2080.04 | 2082.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 14:15:00 | 2081.60 | 2080.04 | 2082.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 2081.60 | 2080.04 | 2082.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 15:00:00 | 2081.60 | 2080.04 | 2082.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 2084.90 | 2081.01 | 2083.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 2064.00 | 2081.01 | 2083.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 2052.60 | 2075.33 | 2080.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 11:00:00 | 2048.20 | 2069.90 | 2077.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 11:45:00 | 2044.50 | 2063.88 | 2073.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 11:15:00 | 1945.79 | 1990.64 | 2019.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 11:15:00 | 1942.27 | 1990.64 | 2019.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 1941.50 | 1935.39 | 1964.59 | SL hit (close>ema200) qty=0.50 sl=1935.39 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 1941.50 | 1935.39 | 1964.59 | SL hit (close>ema200) qty=0.50 sl=1935.39 alert=retest2 |

### Cycle 13 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 1970.90 | 1945.06 | 1941.55 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 1938.30 | 1942.34 | 1942.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 15:15:00 | 1935.00 | 1940.88 | 1942.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 12:15:00 | 1931.90 | 1930.48 | 1935.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 13:00:00 | 1931.90 | 1930.48 | 1935.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 1943.30 | 1933.04 | 1936.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:45:00 | 1941.70 | 1933.04 | 1936.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 1942.40 | 1934.92 | 1936.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 14:30:00 | 1940.40 | 1934.92 | 1936.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 1940.90 | 1936.11 | 1937.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 1922.00 | 1936.11 | 1937.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 1933.60 | 1917.24 | 1916.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 1933.60 | 1917.24 | 1916.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 14:15:00 | 1935.20 | 1920.84 | 1917.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1934.00 | 1950.06 | 1939.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1934.00 | 1950.06 | 1939.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1934.00 | 1950.06 | 1939.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:30:00 | 1931.00 | 1950.06 | 1939.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 1926.50 | 1945.35 | 1937.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 1926.50 | 1945.35 | 1937.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 1907.30 | 1931.77 | 1933.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1902.40 | 1925.90 | 1930.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1844.60 | 1835.81 | 1858.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:45:00 | 1848.70 | 1835.81 | 1858.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1863.20 | 1842.44 | 1854.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 1865.30 | 1842.44 | 1854.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 1864.80 | 1846.91 | 1855.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 12:45:00 | 1858.00 | 1851.69 | 1856.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 13:15:00 | 1857.40 | 1851.69 | 1856.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 14:00:00 | 1857.50 | 1852.85 | 1856.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-01 11:15:00 | 1765.10 | 1787.58 | 1807.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-01 11:15:00 | 1764.53 | 1787.58 | 1807.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-01 11:15:00 | 1764.62 | 1787.58 | 1807.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 14:15:00 | 1791.60 | 1786.20 | 1801.97 | SL hit (close>ema200) qty=0.50 sl=1786.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 14:15:00 | 1791.60 | 1786.20 | 1801.97 | SL hit (close>ema200) qty=0.50 sl=1786.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 14:15:00 | 1791.60 | 1786.20 | 1801.97 | SL hit (close>ema200) qty=0.50 sl=1786.20 alert=retest2 |

### Cycle 17 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 1829.60 | 1793.04 | 1788.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 1838.50 | 1802.13 | 1792.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 12:15:00 | 1900.00 | 1903.58 | 1876.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:00:00 | 1900.00 | 1903.58 | 1876.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1909.70 | 1922.05 | 1907.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:30:00 | 1895.40 | 1922.05 | 1907.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 1918.60 | 1921.36 | 1908.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 10:30:00 | 1907.00 | 1921.36 | 1908.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 14:15:00 | 1914.90 | 1921.13 | 1912.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 15:00:00 | 1914.90 | 1921.13 | 1912.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 1971.70 | 1971.13 | 1957.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 14:30:00 | 1973.50 | 1970.49 | 1962.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 10:00:00 | 1980.00 | 1972.01 | 1964.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 09:15:00 | 1900.00 | 1962.84 | 1965.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-21 09:15:00 | 1900.00 | 1962.84 | 1965.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 09:15:00 | 1900.00 | 1962.84 | 1965.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 1896.00 | 1917.94 | 1936.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1795.30 | 1793.28 | 1826.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:30:00 | 1784.90 | 1793.28 | 1826.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 1816.40 | 1808.79 | 1819.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 10:00:00 | 1816.40 | 1808.79 | 1819.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 1822.10 | 1811.45 | 1819.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:00:00 | 1822.10 | 1811.45 | 1819.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 1817.70 | 1812.70 | 1819.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 15:00:00 | 1808.30 | 1813.07 | 1818.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 09:15:00 | 1826.10 | 1814.96 | 1818.18 | SL hit (close>static) qty=1.00 sl=1823.70 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 12:15:00 | 1811.80 | 1815.91 | 1818.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:15:00 | 1811.20 | 1816.73 | 1818.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:45:00 | 1810.70 | 1815.53 | 1817.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 1819.00 | 1816.22 | 1817.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 1795.00 | 1816.22 | 1817.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1837.10 | 1816.99 | 1815.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1837.10 | 1816.99 | 1815.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1837.10 | 1816.99 | 1815.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1837.10 | 1816.99 | 1815.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 1837.10 | 1816.99 | 1815.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 11:15:00 | 1853.00 | 1835.04 | 1827.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 14:15:00 | 1871.10 | 1872.22 | 1862.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 15:00:00 | 1871.10 | 1872.22 | 1862.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-02-01 11:30:00 | 1980.70 | 2026-02-03 09:15:00 | 2051.70 | STOP_HIT | 1.00 | -3.58% |
| SELL | retest2 | 2026-02-01 12:00:00 | 1963.00 | 2026-02-03 09:15:00 | 2051.70 | STOP_HIT | 1.00 | -4.52% |
| SELL | retest2 | 2026-02-01 14:45:00 | 1976.20 | 2026-02-03 09:15:00 | 2051.70 | STOP_HIT | 1.00 | -3.82% |
| SELL | retest2 | 2026-02-02 10:45:00 | 1980.50 | 2026-02-03 09:15:00 | 2051.70 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2026-02-13 14:00:00 | 2031.90 | 2026-02-25 14:15:00 | 2072.10 | STOP_HIT | 1.00 | 1.98% |
| SELL | retest2 | 2026-02-27 11:00:00 | 2048.20 | 2026-03-04 11:15:00 | 1945.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 11:45:00 | 2044.50 | 2026-03-04 11:15:00 | 1942.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 11:00:00 | 2048.20 | 2026-03-05 14:15:00 | 1941.50 | STOP_HIT | 0.50 | 5.21% |
| SELL | retest2 | 2026-02-27 11:45:00 | 2044.50 | 2026-03-05 14:15:00 | 1941.50 | STOP_HIT | 0.50 | 5.04% |
| SELL | retest2 | 2026-03-13 09:15:00 | 1922.00 | 2026-03-17 13:15:00 | 1933.60 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2026-03-25 12:45:00 | 1858.00 | 2026-04-01 11:15:00 | 1765.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 13:15:00 | 1857.40 | 2026-04-01 11:15:00 | 1764.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 14:00:00 | 1857.50 | 2026-04-01 11:15:00 | 1764.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 12:45:00 | 1858.00 | 2026-04-01 14:15:00 | 1791.60 | STOP_HIT | 0.50 | 3.57% |
| SELL | retest2 | 2026-03-25 13:15:00 | 1857.40 | 2026-04-01 14:15:00 | 1791.60 | STOP_HIT | 0.50 | 3.54% |
| SELL | retest2 | 2026-03-25 14:00:00 | 1857.50 | 2026-04-01 14:15:00 | 1791.60 | STOP_HIT | 0.50 | 3.55% |
| BUY | retest2 | 2026-04-17 14:30:00 | 1973.50 | 2026-04-21 09:15:00 | 1900.00 | STOP_HIT | 1.00 | -3.72% |
| BUY | retest2 | 2026-04-20 10:00:00 | 1980.00 | 2026-04-21 09:15:00 | 1900.00 | STOP_HIT | 1.00 | -4.04% |
| SELL | retest2 | 2026-04-28 15:00:00 | 1808.30 | 2026-04-29 09:15:00 | 1826.10 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2026-04-29 12:15:00 | 1811.80 | 2026-05-04 09:15:00 | 1837.10 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-04-29 14:15:00 | 1811.20 | 2026-05-04 09:15:00 | 1837.10 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2026-04-29 14:45:00 | 1810.70 | 2026-05-04 09:15:00 | 1837.10 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-04-30 09:15:00 | 1795.00 | 2026-05-04 09:15:00 | 1837.10 | STOP_HIT | 1.00 | -2.35% |
