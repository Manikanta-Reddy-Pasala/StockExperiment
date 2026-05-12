# BAJAJFINSV (BAJAJFINSV)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
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
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 2010.90 | 1963.58 | 1976.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 2010.90 | 1963.58 | 1976.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 2010.90 | 1963.58 | 1976.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:45:00 | 2009.00 | 1963.58 | 1976.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 2003.90 | 1963.98 | 1976.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 13:45:00 | 2000.40 | 1965.19 | 1976.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 14:30:00 | 2000.10 | 1965.44 | 1976.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 15:00:00 | 2000.90 | 1960.54 | 1969.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
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
