# BAJAJFINSV (BAJAJFINSV)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1814.00
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
| ALERT2_SKIP | 3 |
| ALERT3 | 43 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 32 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 32 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 35 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 10 / 25
- **Target hits / Stop hits / Partials:** 0 / 31 / 4
- **Avg / median % per leg:** -0.02% / -0.83%
- **Sum % (uncompounded):** -0.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 2 | 8.7% | 0 | 23 | 0 | -1.01% | -23.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 23 | 2 | 8.7% | 0 | 23 | 0 | -1.01% | -23.2% |
| SELL (all) | 12 | 8 | 66.7% | 0 | 8 | 4 | 1.89% | 22.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 8 | 66.7% | 0 | 8 | 4 | 1.89% | 22.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 35 | 10 | 28.6% | 0 | 31 | 4 | -0.02% | -0.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 15:15:00 | 1942.00 | 1992.10 | 1992.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 1928.90 | 1991.47 | 1992.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 2010.90 | 1963.58 | 1976.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 2010.90 | 1963.58 | 1976.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 2010.90 | 1963.58 | 1976.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:45:00 | 2009.00 | 1963.58 | 1976.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 2003.90 | 1963.98 | 1976.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 13:45:00 | 2000.40 | 1965.19 | 1976.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 14:30:00 | 2000.10 | 1965.44 | 1976.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 15:00:00 | 2000.90 | 1960.54 | 1969.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 13:15:00 | 2017.60 | 1963.44 | 1971.11 | SL hit (close>static) qty=1.00 sl=2017.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-05 13:15:00 | 2017.60 | 1963.44 | 1971.11 | SL hit (close>static) qty=1.00 sl=2017.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-05 13:15:00 | 2017.60 | 1963.44 | 1971.11 | SL hit (close>static) qty=1.00 sl=2017.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 11:15:00 | 2036.30 | 1978.32 | 1978.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 12:15:00 | 2038.20 | 1978.92 | 1978.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 2015.00 | 2025.48 | 2006.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-26 10:00:00 | 2015.00 | 2025.48 | 2006.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 2001.40 | 2025.09 | 2006.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 12:00:00 | 2001.40 | 2025.09 | 2006.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 12:15:00 | 2004.50 | 2024.89 | 2006.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 09:30:00 | 2017.40 | 2024.05 | 2006.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 10:15:00 | 2012.70 | 2024.05 | 2006.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 11:15:00 | 1999.30 | 2023.59 | 2006.07 | SL hit (close<static) qty=1.00 sl=2000.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 11:15:00 | 1999.30 | 2023.59 | 2006.07 | SL hit (close<static) qty=1.00 sl=2000.50 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 13:30:00 | 2009.90 | 2023.27 | 2006.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 15:00:00 | 2025.10 | 2023.28 | 2006.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 2008.80 | 2022.64 | 2006.36 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 1983.50 | 2021.89 | 2006.22 | SL hit (close<static) qty=1.00 sl=2000.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 1983.50 | 2021.89 | 2006.22 | SL hit (close<static) qty=1.00 sl=2000.50 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 09:15:00 | 2021.40 | 2018.69 | 2005.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 12:45:00 | 2019.10 | 2020.72 | 2007.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 10:30:00 | 2016.20 | 2020.33 | 2007.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 12:30:00 | 2013.00 | 2020.03 | 2008.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 2007.30 | 2019.90 | 2008.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:00:00 | 2007.30 | 2019.90 | 2008.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 2003.40 | 2019.74 | 2008.25 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-10 14:15:00 | 2003.40 | 2019.74 | 2008.25 | SL hit (close<static) qty=1.00 sl=2004.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-10 14:15:00 | 2003.40 | 2019.74 | 2008.25 | SL hit (close<static) qty=1.00 sl=2004.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-10 14:15:00 | 2003.40 | 2019.74 | 2008.25 | SL hit (close<static) qty=1.00 sl=2004.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-10 14:15:00 | 2003.40 | 2019.74 | 2008.25 | SL hit (close<static) qty=1.00 sl=2004.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-10-10 14:45:00 | 2007.40 | 2019.74 | 2008.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 2008.50 | 2019.63 | 2008.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 2006.00 | 2019.45 | 2008.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 2009.30 | 2019.35 | 2008.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 11:30:00 | 2013.90 | 2019.29 | 2008.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 13:00:00 | 2015.30 | 2019.26 | 2008.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 1978.70 | 2079.64 | 2053.12 | SL hit (close<static) qty=1.00 sl=1998.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 1978.70 | 2079.64 | 2053.12 | SL hit (close<static) qty=1.00 sl=1998.30 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 10:30:00 | 2013.10 | 2072.41 | 2050.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 11:45:00 | 2015.90 | 2071.79 | 2050.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 2045.00 | 2069.29 | 2050.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:00:00 | 2045.00 | 2069.29 | 2050.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 2054.20 | 2069.14 | 2050.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:15:00 | 2055.00 | 2069.00 | 2050.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 13:45:00 | 2055.20 | 2068.72 | 2050.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 12:30:00 | 2056.30 | 2068.29 | 2051.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 14:00:00 | 2058.80 | 2068.19 | 2051.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 2050.00 | 2068.01 | 2051.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:00:00 | 2050.00 | 2068.01 | 2051.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 2053.30 | 2067.87 | 2051.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:15:00 | 2038.60 | 2067.87 | 2051.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 2042.90 | 2067.62 | 2051.25 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 2042.90 | 2067.62 | 2051.25 | SL hit (close<static) qty=1.00 sl=2044.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 2042.90 | 2067.62 | 2051.25 | SL hit (close<static) qty=1.00 sl=2044.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 2042.90 | 2067.62 | 2051.25 | SL hit (close<static) qty=1.00 sl=2044.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 2042.90 | 2067.62 | 2051.25 | SL hit (close<static) qty=1.00 sl=2044.20 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-11-19 09:45:00 | 2038.90 | 2067.62 | 2051.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 2039.10 | 2067.33 | 2051.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:00:00 | 2039.10 | 2067.33 | 2051.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 2049.20 | 2066.29 | 2051.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 2051.50 | 2066.29 | 2051.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 2057.30 | 2066.20 | 2051.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 10:45:00 | 2065.70 | 2066.26 | 2051.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 11:15:00 | 2038.60 | 2067.28 | 2052.84 | SL hit (close<static) qty=1.00 sl=2042.90 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 12:00:00 | 2068.50 | 2064.39 | 2052.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 10:15:00 | 2066.70 | 2070.72 | 2057.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 14:15:00 | 2065.00 | 2070.65 | 2057.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 2044.50 | 2070.31 | 2057.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 2044.50 | 2070.31 | 2057.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 2038.90 | 2070.00 | 2057.34 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 2038.90 | 2070.00 | 2057.34 | SL hit (close<static) qty=1.00 sl=2042.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 2038.90 | 2070.00 | 2057.34 | SL hit (close<static) qty=1.00 sl=2042.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 2038.90 | 2070.00 | 2057.34 | SL hit (close<static) qty=1.00 sl=2042.90 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-12-03 11:00:00 | 2038.90 | 2070.00 | 2057.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 2050.80 | 2068.37 | 2057.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:45:00 | 2050.20 | 2068.37 | 2057.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 2044.30 | 2068.13 | 2056.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:30:00 | 2043.30 | 2068.13 | 2056.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 2052.60 | 2069.27 | 2058.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:00:00 | 2052.60 | 2069.27 | 2058.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 2061.00 | 2069.19 | 2058.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:45:00 | 2042.50 | 2069.19 | 2058.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 15:15:00 | 2057.10 | 2069.07 | 2058.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:15:00 | 2048.50 | 2069.07 | 2058.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 2047.90 | 2068.86 | 2058.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:45:00 | 2044.10 | 2068.86 | 2058.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 2064.90 | 2069.28 | 2059.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:45:00 | 2058.40 | 2069.28 | 2059.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 2060.50 | 2069.19 | 2059.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:00:00 | 2060.50 | 2069.19 | 2059.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 2062.40 | 2069.13 | 2059.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:45:00 | 2060.00 | 2069.13 | 2059.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 2057.10 | 2069.01 | 2059.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:00:00 | 2057.10 | 2069.01 | 2059.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 2066.30 | 2068.98 | 2059.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 14:15:00 | 2069.60 | 2068.98 | 2059.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:15:00 | 2076.70 | 2068.91 | 2059.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 11:15:00 | 2068.90 | 2069.55 | 2060.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 2046.90 | 2069.43 | 2060.31 | SL hit (close<static) qty=1.00 sl=2051.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 2046.90 | 2069.43 | 2060.31 | SL hit (close<static) qty=1.00 sl=2051.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 2046.90 | 2069.43 | 2060.31 | SL hit (close<static) qty=1.00 sl=2051.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 12:15:00 | 2009.50 | 2053.54 | 2053.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 12:15:00 | 2009.50 | 2053.54 | 2053.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 2009.50 | 2053.54 | 2053.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 1988.60 | 2051.66 | 2052.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 09:15:00 | 2073.20 | 2046.18 | 2049.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 2073.20 | 2046.18 | 2049.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 2073.20 | 2046.18 | 2049.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:00:00 | 2073.20 | 2046.18 | 2049.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 2062.40 | 2046.34 | 2049.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:30:00 | 2074.20 | 2046.34 | 2049.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 2049.70 | 2046.40 | 2049.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 14:15:00 | 2043.80 | 2046.40 | 2049.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 15:00:00 | 2044.50 | 2046.38 | 2049.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 09:15:00 | 2023.10 | 2046.38 | 2049.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 1941.61 | 2024.09 | 2035.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 1942.27 | 2024.09 | 2035.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 13:15:00 | 1921.94 | 2010.93 | 2027.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 2010.50 | 1986.60 | 2011.45 | SL hit (close>ema200) qty=0.50 sl=1986.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 2010.50 | 1986.60 | 2011.45 | SL hit (close>ema200) qty=0.50 sl=1986.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 2010.50 | 1986.60 | 2011.45 | SL hit (close>ema200) qty=0.50 sl=1986.60 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 11:45:00 | 2039.00 | 1997.65 | 2012.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 2033.00 | 1999.41 | 2013.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:30:00 | 2030.20 | 1999.41 | 2013.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 2017.10 | 2001.97 | 2013.76 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-16 15:15:00 | 2053.10 | 2005.46 | 2014.97 | SL hit (close>static) qty=1.00 sl=2052.00 alert=retest2 |

### Cycle 4 — BUY (started 2026-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 14:15:00 | 2049.70 | 2022.50 | 2022.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 2052.80 | 2023.00 | 2022.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 2016.60 | 2024.11 | 2023.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 2016.60 | 2024.11 | 2023.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 2016.60 | 2024.11 | 2023.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:45:00 | 2016.50 | 2024.11 | 2023.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 2020.60 | 2024.07 | 2023.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:15:00 | 2016.10 | 2024.07 | 2023.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 2017.00 | 2024.00 | 2023.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:30:00 | 2013.10 | 2024.00 | 2023.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 1966.20 | 2022.51 | 2022.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 1946.20 | 2021.16 | 2021.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 1805.00 | 1783.43 | 1865.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 11:00:00 | 1805.00 | 1783.43 | 1865.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 1852.40 | 1797.68 | 1852.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 12:45:00 | 1852.00 | 1797.68 | 1852.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 1849.40 | 1798.19 | 1852.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 14:00:00 | 1849.40 | 1798.19 | 1852.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 1850.40 | 1798.71 | 1852.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 15:15:00 | 1850.10 | 1798.71 | 1852.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 1850.10 | 1799.22 | 1852.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:15:00 | 1854.00 | 1799.22 | 1852.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 1840.80 | 1799.64 | 1852.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 09:15:00 | 1827.80 | 1802.48 | 1852.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 14:15:00 | 1736.41 | 1793.15 | 1838.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 12:15:00 | 1791.30 | 1790.06 | 1834.08 | SL hit (close>ema200) qty=0.50 sl=1790.06 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 09:45:00 | 1833.70 | 1792.44 | 1832.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-08-18 13:45:00 | 2000.40 | 2025-09-05 13:15:00 | 2017.60 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-08-18 14:30:00 | 2000.10 | 2025-09-05 13:15:00 | 2017.60 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-09-04 15:00:00 | 2000.90 | 2025-09-05 13:15:00 | 2017.60 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-09-29 09:30:00 | 2017.40 | 2025-09-29 11:15:00 | 1999.30 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-09-29 10:15:00 | 2012.70 | 2025-09-29 11:15:00 | 1999.30 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-09-29 13:30:00 | 2009.90 | 2025-10-01 09:15:00 | 1983.50 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-09-29 15:00:00 | 2025.10 | 2025-10-01 09:15:00 | 1983.50 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-10-06 09:15:00 | 2021.40 | 2025-10-10 14:15:00 | 2003.40 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-10-08 12:45:00 | 2019.10 | 2025-10-10 14:15:00 | 2003.40 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-10-09 10:30:00 | 2016.20 | 2025-10-10 14:15:00 | 2003.40 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-10-10 12:30:00 | 2013.00 | 2025-10-10 14:15:00 | 2003.40 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-10-13 11:30:00 | 2013.90 | 2025-11-11 09:15:00 | 1978.70 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-10-14 13:00:00 | 2015.30 | 2025-11-11 09:15:00 | 1978.70 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-11-12 10:30:00 | 2013.10 | 2025-11-19 09:15:00 | 2042.90 | STOP_HIT | 1.00 | 1.48% |
| BUY | retest2 | 2025-11-12 11:45:00 | 2015.90 | 2025-11-19 09:15:00 | 2042.90 | STOP_HIT | 1.00 | 1.34% |
| BUY | retest2 | 2025-11-14 12:15:00 | 2055.00 | 2025-11-19 09:15:00 | 2042.90 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-11-14 13:45:00 | 2055.20 | 2025-11-19 09:15:00 | 2042.90 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-11-18 12:30:00 | 2056.30 | 2025-11-24 11:15:00 | 2038.60 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-11-18 14:00:00 | 2058.80 | 2025-12-03 10:15:00 | 2038.90 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-11-20 10:45:00 | 2065.70 | 2025-12-03 10:15:00 | 2038.90 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-11-26 12:00:00 | 2068.50 | 2025-12-03 10:15:00 | 2038.90 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-12-02 10:15:00 | 2066.70 | 2025-12-16 09:15:00 | 2046.90 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-12-02 14:15:00 | 2065.00 | 2025-12-16 09:15:00 | 2046.90 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-12-11 14:15:00 | 2069.60 | 2025-12-16 09:15:00 | 2046.90 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-12-12 09:15:00 | 2076.70 | 2025-12-29 12:15:00 | 2009.50 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2025-12-15 11:15:00 | 2068.90 | 2025-12-29 12:15:00 | 2009.50 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2026-01-06 14:15:00 | 2043.80 | 2026-01-21 10:15:00 | 1941.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 15:00:00 | 2044.50 | 2026-01-21 10:15:00 | 1942.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 09:15:00 | 2023.10 | 2026-01-27 13:15:00 | 1921.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 14:15:00 | 2043.80 | 2026-02-03 09:15:00 | 2010.50 | STOP_HIT | 0.50 | 1.63% |
| SELL | retest2 | 2026-01-06 15:00:00 | 2044.50 | 2026-02-03 09:15:00 | 2010.50 | STOP_HIT | 0.50 | 1.66% |
| SELL | retest2 | 2026-01-07 09:15:00 | 2023.10 | 2026-02-03 09:15:00 | 2010.50 | STOP_HIT | 0.50 | 0.62% |
| SELL | retest2 | 2026-02-11 11:45:00 | 2039.00 | 2026-02-16 15:15:00 | 2053.10 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2026-04-23 09:15:00 | 1827.80 | 2026-04-30 14:15:00 | 1736.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-23 09:15:00 | 1827.80 | 2026-05-05 12:15:00 | 1791.30 | STOP_HIT | 0.50 | 2.00% |
