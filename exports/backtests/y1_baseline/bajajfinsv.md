# BAJAJFINSV (BAJAJFINSV)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
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
| ALERT2_SKIP | 4 |
| ALERT3 | 8 |
| PENDING | 27 |
| PENDING_CANCEL | 7 |
| ENTRY1 | 2 |
| ENTRY2 | 18 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 2 / 18
- **Target hits / Stop hits / Partials:** 0 / 19 / 1
- **Avg / median % per leg:** -1.76% / -1.67%
- **Sum % (uncompounded):** -35.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 0 | 0.0% | 0 | 13 | 0 | -2.03% | -26.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 0 | 0.0% | 0 | 13 | 0 | -2.03% | -26.5% |
| SELL (all) | 7 | 2 | 28.6% | 0 | 6 | 1 | -1.26% | -8.8% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -5.00% | -10.0% |
| SELL @ 3rd Alert (retest2) | 5 | 2 | 40.0% | 0 | 4 | 1 | 0.24% | 1.2% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -5.00% | -10.0% |
| retest2 (combined) | 18 | 2 | 11.1% | 0 | 17 | 1 | -1.40% | -25.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 11:15:00 | 1958.10 | 2012.34 | 2012.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 1945.70 | 2009.60 | 2011.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 2010.90 | 1963.99 | 1983.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 2010.90 | 1963.99 | 1983.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 2010.90 | 1963.99 | 1983.76 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-08-26 09:15:00 | 1939.90 | 1966.41 | 1981.45 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 10:15:00 | 1943.40 | 1966.18 | 1981.26 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-02 13:15:00 | 1949.20 | 1957.30 | 1974.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-02 14:15:00 | 1957.70 | 1957.31 | 1974.19 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-09-04 12:15:00 | 2016.20 | 1959.82 | 1974.47 | SL hit (close>static) qty=1.00 sl=2015.90 alert=retest2 |

### Cycle 2 — BUY (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 15:15:00 | 2083.20 | 1986.61 | 1986.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 2102.90 | 1994.14 | 1990.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 2015.00 | 2025.53 | 2009.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 10:15:00 | 2010.80 | 2025.39 | 2009.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 2010.80 | 2025.39 | 2009.04 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-09-29 14:15:00 | 2025.10 | 2023.34 | 2008.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-29 15:15:00 | 2012.00 | 2023.22 | 2008.88 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-06 09:15:00 | 2025.00 | 2018.79 | 2008.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 10:15:00 | 2021.80 | 2018.82 | 2008.12 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-09 10:15:00 | 2020.20 | 2020.36 | 2010.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 11:15:00 | 2022.00 | 2020.38 | 2010.08 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-10 10:15:00 | 2024.00 | 2020.21 | 2010.30 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-10 11:15:00 | 2015.20 | 2020.16 | 2010.32 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-10-10 13:15:00 | 2007.30 | 2019.93 | 2010.31 | SL hit (close<static) qty=1.00 sl=2008.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-10 13:15:00 | 2007.30 | 2019.93 | 2010.31 | SL hit (close<static) qty=1.00 sl=2008.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-13 15:15:00 | 2023.90 | 2019.26 | 2010.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 09:15:00 | 2025.30 | 2019.32 | 2010.46 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2025-10-15 09:15:00 | 2053.80 | 2019.58 | 2010.89 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 10:15:00 | 2061.90 | 2020.00 | 2011.14 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 2052.00 | 2077.87 | 2051.48 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-11-07 11:15:00 | 2072.60 | 2077.58 | 2051.59 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 12:15:00 | 2081.90 | 2077.62 | 2051.74 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 1978.70 | 2079.64 | 2054.18 | SL hit (close<static) qty=1.00 sl=2008.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 1978.70 | 2079.64 | 2054.18 | SL hit (close<static) qty=1.00 sl=2008.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 1978.70 | 2079.64 | 2054.18 | SL hit (close<static) qty=1.00 sl=2042.80 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-14 14:15:00 | 2068.70 | 2068.72 | 2051.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 15:15:00 | 2064.90 | 2068.69 | 2051.40 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-17 12:15:00 | 2067.90 | 2068.42 | 2051.61 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 13:15:00 | 2074.30 | 2068.48 | 2051.72 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-19 10:15:00 | 2039.10 | 2067.34 | 2052.04 | SL hit (close<static) qty=1.00 sl=2042.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 10:15:00 | 2039.10 | 2067.34 | 2052.04 | SL hit (close<static) qty=1.00 sl=2042.80 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-20 10:15:00 | 2072.50 | 2066.27 | 2052.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 11:15:00 | 2079.30 | 2066.39 | 2052.16 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 2053.00 | 2067.82 | 2053.59 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-24 11:15:00 | 2038.60 | 2067.28 | 2053.60 | SL hit (close<static) qty=1.00 sl=2042.80 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-26 12:15:00 | 2076.00 | 2064.51 | 2053.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 13:15:00 | 2077.00 | 2064.64 | 2053.26 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 2044.50 | 2070.31 | 2058.04 | SL hit (close<static) qty=1.00 sl=2045.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-05 09:15:00 | 2081.60 | 2067.68 | 2057.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 10:15:00 | 2093.30 | 2067.94 | 2057.68 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-08 10:15:00 | 2073.90 | 2069.47 | 2058.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-08 11:15:00 | 2069.60 | 2069.48 | 2058.87 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-09 11:15:00 | 2078.80 | 2068.89 | 2058.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 12:15:00 | 2072.30 | 2068.92 | 2059.00 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-10 09:15:00 | 2089.00 | 2069.14 | 2059.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 10:15:00 | 2078.90 | 2069.24 | 2059.41 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 2064.90 | 2069.28 | 2059.72 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-12-12 12:15:00 | 2083.00 | 2069.18 | 2060.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 13:15:00 | 2086.00 | 2069.34 | 2060.26 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-12 15:15:00 | 2084.30 | 2069.62 | 2060.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-15 09:15:00 | 2066.90 | 2069.59 | 2060.52 | ENTRY2 sustain failed after 3960m |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 2046.90 | 2069.43 | 2060.75 | SL hit (close<static) qty=1.00 sl=2052.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-16 12:15:00 | 2044.20 | 2068.88 | 2060.60 | SL hit (close<static) qty=1.00 sl=2045.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-16 12:15:00 | 2044.20 | 2068.88 | 2060.60 | SL hit (close<static) qty=1.00 sl=2045.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-16 12:15:00 | 2044.20 | 2068.88 | 2060.60 | SL hit (close<static) qty=1.00 sl=2045.10 alert=retest2 |

### Cycle 3 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 2006.30 | 2053.98 | 2054.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 1988.60 | 2051.66 | 2053.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 09:15:00 | 2073.20 | 2046.18 | 2049.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 2073.20 | 2046.18 | 2049.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 2073.20 | 2046.18 | 2049.71 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-01-07 09:15:00 | 2030.50 | 2046.22 | 2049.61 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 10:15:00 | 2031.30 | 2046.07 | 2049.52 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 12:15:00 | 1929.73 | 2011.99 | 2028.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 2010.50 | 1986.60 | 2011.59 | SL hit (close>ema200) qty=0.50 sl=1986.60 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-11 11:15:00 | 2032.40 | 1997.65 | 2012.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 12:15:00 | 2030.00 | 1997.97 | 2012.79 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-19 13:15:00 | 2040.30 | 2013.25 | 2018.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 14:15:00 | 2033.60 | 2013.45 | 2018.35 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-24 09:15:00 | 2032.90 | 2018.98 | 2020.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-24 10:15:00 | 2042.00 | 2019.20 | 2020.95 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-24 15:15:00 | 2038.20 | 2020.31 | 2021.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-25 09:15:00 | 2071.40 | 2020.82 | 2021.71 | ENTRY2 sustain failed after 1080m |
| Stop hit — per-position SL triggered | 2026-02-25 15:15:00 | 2043.00 | 2022.70 | 2022.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 15:15:00 | 2043.00 | 2022.70 | 2022.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2026-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 15:15:00 | 2043.00 | 2022.70 | 2022.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 2052.80 | 2023.00 | 2022.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 2016.60 | 2024.11 | 2023.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 2016.60 | 2024.11 | 2023.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 2016.60 | 2024.11 | 2023.37 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 1966.20 | 2022.51 | 2022.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 1946.20 | 2021.16 | 2021.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 1805.00 | 1783.43 | 1865.98 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-09 12:15:00 | 1775.00 | 1783.29 | 1862.29 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 13:15:00 | 1763.00 | 1783.09 | 1861.79 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 1769.60 | 1784.18 | 1858.50 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1774.40 | 1784.08 | 1858.08 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 1852.40 | 1797.68 | 1852.89 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-22 12:15:00 | 1857.20 | 1801.13 | 1852.75 | SL hit (close>ema400) qty=1.00 sl=1852.75 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-22 12:15:00 | 1857.20 | 1801.13 | 1852.75 | SL hit (close>ema400) qty=1.00 sl=1852.75 alert=retest1 |
| Cross detected — sustain check pending | 2026-04-23 09:15:00 | 1818.00 | 1802.64 | 1852.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:15:00 | 1796.20 | 1802.57 | 1852.21 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-08-26 10:15:00 | 1943.40 | 2025-09-04 12:15:00 | 2016.20 | STOP_HIT | 1.00 | -3.75% |
| BUY | retest2 | 2025-10-06 10:15:00 | 2021.80 | 2025-10-10 13:15:00 | 2007.30 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-10-09 11:15:00 | 2022.00 | 2025-10-10 13:15:00 | 2007.30 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-10-14 09:15:00 | 2025.30 | 2025-11-11 09:15:00 | 1978.70 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-10-15 10:15:00 | 2061.90 | 2025-11-11 09:15:00 | 1978.70 | STOP_HIT | 1.00 | -4.04% |
| BUY | retest2 | 2025-11-07 12:15:00 | 2081.90 | 2025-11-11 09:15:00 | 1978.70 | STOP_HIT | 1.00 | -4.96% |
| BUY | retest2 | 2025-11-14 15:15:00 | 2064.90 | 2025-11-19 10:15:00 | 2039.10 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-11-17 13:15:00 | 2074.30 | 2025-11-19 10:15:00 | 2039.10 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-11-20 11:15:00 | 2079.30 | 2025-11-24 11:15:00 | 2038.60 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-11-26 13:15:00 | 2077.00 | 2025-12-03 09:15:00 | 2044.50 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-12-05 10:15:00 | 2093.30 | 2025-12-16 09:15:00 | 2046.90 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-12-09 12:15:00 | 2072.30 | 2025-12-16 12:15:00 | 2044.20 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-12-10 10:15:00 | 2078.90 | 2025-12-16 12:15:00 | 2044.20 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-12-12 13:15:00 | 2086.00 | 2025-12-16 12:15:00 | 2044.20 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2026-01-07 10:15:00 | 2031.30 | 2026-01-27 12:15:00 | 1929.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 10:15:00 | 2031.30 | 2026-02-03 09:15:00 | 2010.50 | STOP_HIT | 0.50 | 1.02% |
| SELL | retest2 | 2026-02-11 12:15:00 | 2030.00 | 2026-02-25 15:15:00 | 2043.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2026-02-19 14:15:00 | 2033.60 | 2026-02-25 15:15:00 | 2043.00 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-04-09 13:15:00 | 1763.00 | 2026-04-22 12:15:00 | 1857.20 | STOP_HIT | 1.00 | -5.34% |
| SELL | retest1 | 2026-04-13 10:15:00 | 1774.40 | 2026-04-22 12:15:00 | 1857.20 | STOP_HIT | 1.00 | -4.67% |
