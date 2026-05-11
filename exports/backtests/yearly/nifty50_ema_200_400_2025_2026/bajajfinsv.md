# BAJAJFINSV (BAJAJFINSV)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3164 bars)
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
| ALERT2 | 6 |
| ALERT2_SKIP | 3 |
| ALERT3 | 22 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 18 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 8 / 13
- **Target hits / Stop hits / Partials:** 0 / 17 / 4
- **Avg / median % per leg:** 0.45% / -0.83%
- **Sum % (uncompounded):** 9.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.47% | -13.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.47% | -13.2% |
| SELL (all) | 12 | 8 | 66.7% | 0 | 8 | 4 | 1.89% | 22.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 8 | 66.7% | 0 | 8 | 4 | 1.89% | 22.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 21 | 8 | 38.1% | 0 | 17 | 4 | 0.45% | 9.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 14:15:00 | 1908.30 | 1969.38 | 1969.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 15:15:00 | 1905.00 | 1968.74 | 1969.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 2010.90 | 1962.84 | 1965.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 2010.90 | 1962.84 | 1965.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 2010.90 | 1962.84 | 1965.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:45:00 | 2009.00 | 1962.84 | 1965.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 2003.90 | 1963.25 | 1966.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 13:45:00 | 2000.40 | 1964.48 | 1966.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 14:30:00 | 2000.10 | 1964.74 | 1966.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 15:00:00 | 2000.90 | 1960.24 | 1963.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 13:15:00 | 2017.60 | 1963.15 | 1964.77 | SL hit (close>static) qty=1.00 sl=2017.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 13:15:00 | 2019.00 | 1966.83 | 1966.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 09:15:00 | 2034.90 | 1968.33 | 1967.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 2015.00 | 2025.37 | 2002.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-26 10:00:00 | 2015.00 | 2025.37 | 2002.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 2001.40 | 2024.99 | 2002.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 12:00:00 | 2001.40 | 2024.99 | 2002.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 12:15:00 | 2004.50 | 2024.79 | 2002.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 09:30:00 | 2017.40 | 2023.95 | 2002.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 10:15:00 | 2012.70 | 2023.95 | 2002.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 11:15:00 | 1999.30 | 2023.49 | 2002.41 | SL hit (close<static) qty=1.00 sl=2000.50 alert=retest2 |

### Cycle 3 — SELL (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 14:15:00 | 2011.90 | 2052.69 | 2052.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 1988.60 | 2051.66 | 2052.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 09:15:00 | 2073.20 | 2046.18 | 2049.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 2073.20 | 2046.18 | 2049.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 2073.20 | 2046.18 | 2049.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:00:00 | 2073.20 | 2046.18 | 2049.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 2062.40 | 2046.34 | 2049.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:30:00 | 2074.20 | 2046.34 | 2049.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 2049.70 | 2046.40 | 2049.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 14:15:00 | 2043.80 | 2046.40 | 2049.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 15:00:00 | 2044.50 | 2046.38 | 2049.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 09:15:00 | 2023.10 | 2046.37 | 2049.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 1941.61 | 2024.09 | 2035.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 1942.27 | 2024.09 | 2035.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 13:15:00 | 1921.94 | 2010.93 | 2027.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 2010.50 | 1986.60 | 2011.27 | SL hit (close>ema200) qty=0.50 sl=1986.60 alert=retest2 |

### Cycle 4 — BUY (started 2026-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 14:15:00 | 2049.70 | 2022.50 | 2022.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 2052.80 | 2023.00 | 2022.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 2016.60 | 2024.11 | 2023.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 2016.60 | 2024.11 | 2023.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 2016.60 | 2024.11 | 2023.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:45:00 | 2016.50 | 2024.11 | 2023.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 2020.60 | 2024.07 | 2023.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:15:00 | 2016.10 | 2024.07 | 2023.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 2017.00 | 2024.00 | 2023.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:30:00 | 2013.10 | 2024.00 | 2023.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 1962.90 | 2021.91 | 2022.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 1946.20 | 2021.16 | 2021.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 1805.00 | 1783.43 | 1865.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 11:00:00 | 1805.00 | 1783.43 | 1865.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 1852.40 | 1797.68 | 1852.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 12:45:00 | 1852.00 | 1797.68 | 1852.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 1849.40 | 1798.19 | 1852.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 14:00:00 | 1849.40 | 1798.19 | 1852.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 1850.40 | 1798.71 | 1852.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 15:15:00 | 1850.10 | 1798.71 | 1852.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 1850.10 | 1799.22 | 1852.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:15:00 | 1854.00 | 1799.22 | 1852.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 1840.80 | 1799.64 | 1852.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 09:15:00 | 1827.80 | 1802.48 | 1852.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 14:15:00 | 1736.41 | 1793.15 | 1838.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 12:15:00 | 1791.30 | 1790.06 | 1834.06 | SL hit (close>ema200) qty=0.50 sl=1790.06 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-25 12:00:00 | 1965.80 | 2025-08-01 14:15:00 | 1913.50 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2025-07-31 10:30:00 | 1949.80 | 2025-08-01 14:15:00 | 1913.50 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-08-01 13:00:00 | 1946.50 | 2025-08-01 14:15:00 | 1913.50 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-08-18 13:45:00 | 2000.40 | 2025-09-05 13:15:00 | 2017.60 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-08-18 14:30:00 | 2000.10 | 2025-09-05 13:15:00 | 2017.60 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-09-04 15:00:00 | 2000.90 | 2025-09-05 13:15:00 | 2017.60 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-09-29 09:30:00 | 2017.40 | 2025-09-29 11:15:00 | 1999.30 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-09-29 10:15:00 | 2012.70 | 2025-09-29 11:15:00 | 1999.30 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-09-29 13:30:00 | 2009.90 | 2025-10-01 09:15:00 | 1983.50 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-09-29 15:00:00 | 2025.10 | 2025-10-01 09:15:00 | 1983.50 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-10-01 11:30:00 | 1996.90 | 2025-10-03 09:15:00 | 1974.20 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-10-03 14:45:00 | 1996.70 | 2025-11-11 09:15:00 | 1978.70 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2026-01-06 14:15:00 | 2043.80 | 2026-01-21 10:15:00 | 1941.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 15:00:00 | 2044.50 | 2026-01-21 10:15:00 | 1942.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 09:15:00 | 2023.10 | 2026-01-27 13:15:00 | 1921.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 14:15:00 | 2043.80 | 2026-02-03 09:15:00 | 2010.50 | STOP_HIT | 0.50 | 1.63% |
| SELL | retest2 | 2026-01-06 15:00:00 | 2044.50 | 2026-02-03 09:15:00 | 2010.50 | STOP_HIT | 0.50 | 1.66% |
| SELL | retest2 | 2026-01-07 09:15:00 | 2023.10 | 2026-02-03 09:15:00 | 2010.50 | STOP_HIT | 0.50 | 0.62% |
| SELL | retest2 | 2026-02-11 11:45:00 | 2039.00 | 2026-02-16 15:15:00 | 2053.10 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2026-04-23 09:15:00 | 1827.80 | 2026-04-30 14:15:00 | 1736.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-23 09:15:00 | 1827.80 | 2026-05-05 12:15:00 | 1791.30 | STOP_HIT | 0.50 | 2.00% |
