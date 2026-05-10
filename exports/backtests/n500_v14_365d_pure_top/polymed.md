# Poly Medicure Ltd. (POLYMED)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1649.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 12 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 16 |
| PARTIAL | 9 |
| TARGET_HIT | 5 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 8
- **Target hits / Stop hits / Partials:** 5 / 11 / 9
- **Avg / median % per leg:** 3.46% / 4.84%
- **Sum % (uncompounded):** 86.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.27% | -15.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.27% | -15.9% |
| SELL (all) | 18 | 17 | 94.4% | 5 | 4 | 9 | 5.69% | 102.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 17 | 94.4% | 5 | 4 | 9 | 5.69% | 102.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 25 | 17 | 68.0% | 5 | 11 | 9 | 3.46% | 86.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 10:15:00 | 2243.60 | 2374.75 | 2375.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 12:15:00 | 2235.10 | 2372.09 | 2373.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 14:15:00 | 2233.10 | 2231.85 | 2284.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-30 15:00:00 | 2233.10 | 2231.85 | 2284.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 2251.40 | 2233.45 | 2283.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 13:15:00 | 2237.40 | 2234.24 | 2282.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 09:30:00 | 2240.80 | 2235.30 | 2282.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:30:00 | 2244.50 | 2235.56 | 2281.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 13:15:00 | 2242.20 | 2235.56 | 2281.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-16 11:15:00 | 2128.76 | 2224.05 | 2263.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-16 11:15:00 | 2132.28 | 2224.05 | 2263.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-16 11:15:00 | 2130.09 | 2224.05 | 2263.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-16 12:15:00 | 2125.53 | 2222.99 | 2262.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-07-28 09:15:00 | 2013.66 | 2164.93 | 2220.94 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-07-28 09:15:00 | 2016.72 | 2164.93 | 2220.94 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-07-28 09:15:00 | 2020.05 | 2164.93 | 2220.94 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-07-28 09:15:00 | 2017.98 | 2164.93 | 2220.94 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 2051.00 | 2003.05 | 2088.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 13:00:00 | 2051.00 | 2003.05 | 2088.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 2087.90 | 2003.89 | 2088.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 13:45:00 | 2080.50 | 2003.89 | 2088.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 2069.90 | 2004.55 | 2088.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 15:00:00 | 2026.50 | 2036.10 | 2086.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 11:45:00 | 2040.00 | 2036.54 | 2085.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 13:15:00 | 1938.00 | 2030.62 | 2077.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 2059.20 | 2019.60 | 2063.15 | SL hit (close>ema200) qty=0.50 sl=2019.60 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 10:15:00 | 2049.00 | 2023.25 | 2063.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-22 13:15:00 | 1946.55 | 2020.74 | 2060.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 2035.00 | 2012.61 | 2052.61 | SL hit (close>ema200) qty=0.50 sl=2012.61 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 10:45:00 | 2048.60 | 2012.90 | 2052.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 12:15:00 | 2050.20 | 2013.57 | 2052.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:30:00 | 2052.60 | 2013.57 | 2052.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 2054.30 | 2013.98 | 2052.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 13:45:00 | 2054.60 | 2013.98 | 2052.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 2030.30 | 2014.14 | 2052.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 15:15:00 | 2029.00 | 2014.14 | 2052.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 10:15:00 | 1946.17 | 2012.98 | 2051.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:15:00 | 1925.17 | 2012.15 | 2050.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:15:00 | 1927.55 | 2012.15 | 2050.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-10-15 09:15:00 | 1843.74 | 1950.15 | 2001.54 | Target hit (10%) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-16 12:15:00 | 1948.70 | 1942.95 | 1995.31 | SL hit (close>ema200) qty=0.50 sl=1942.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-16 12:15:00 | 1948.70 | 1942.95 | 1995.31 | SL hit (close>ema200) qty=0.50 sl=1942.95 alert=retest2 |

### Cycle 2 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 1624.00 | 1447.03 | 1446.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 12:15:00 | 1638.80 | 1448.94 | 1447.26 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-19 13:45:00 | 2439.90 | 2025-05-27 11:15:00 | 2379.80 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2025-05-19 14:15:00 | 2438.40 | 2025-05-27 11:15:00 | 2379.80 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-05-20 14:30:00 | 2439.00 | 2025-05-27 11:15:00 | 2379.80 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2025-05-23 11:00:00 | 2452.90 | 2025-05-27 13:15:00 | 2372.50 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest2 | 2025-05-26 11:45:00 | 2413.20 | 2025-05-27 13:15:00 | 2372.50 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-05-26 13:45:00 | 2410.00 | 2025-05-27 13:15:00 | 2372.50 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-05-26 14:15:00 | 2422.60 | 2025-05-27 13:15:00 | 2372.50 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-07-02 13:15:00 | 2237.40 | 2025-07-16 11:15:00 | 2128.76 | PARTIAL | 0.50 | 4.86% |
| SELL | retest2 | 2025-07-03 09:30:00 | 2240.80 | 2025-07-16 11:15:00 | 2132.28 | PARTIAL | 0.50 | 4.84% |
| SELL | retest2 | 2025-07-03 12:30:00 | 2244.50 | 2025-07-16 11:15:00 | 2130.09 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2025-07-03 13:15:00 | 2242.20 | 2025-07-16 12:15:00 | 2125.53 | PARTIAL | 0.50 | 5.20% |
| SELL | retest2 | 2025-07-02 13:15:00 | 2237.40 | 2025-07-28 09:15:00 | 2013.66 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-03 09:30:00 | 2240.80 | 2025-07-28 09:15:00 | 2016.72 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-03 12:30:00 | 2244.50 | 2025-07-28 09:15:00 | 2020.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-03 13:15:00 | 2242.20 | 2025-07-28 09:15:00 | 2017.98 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-04 15:00:00 | 2026.50 | 2025-09-10 13:15:00 | 1938.00 | PARTIAL | 0.50 | 4.37% |
| SELL | retest2 | 2025-09-04 15:00:00 | 2026.50 | 2025-09-18 09:15:00 | 2059.20 | STOP_HIT | 0.50 | -1.61% |
| SELL | retest2 | 2025-09-05 11:45:00 | 2040.00 | 2025-09-22 13:15:00 | 1946.55 | PARTIAL | 0.50 | 4.58% |
| SELL | retest2 | 2025-09-05 11:45:00 | 2040.00 | 2025-09-25 09:15:00 | 2035.00 | STOP_HIT | 0.50 | 0.25% |
| SELL | retest2 | 2025-09-19 10:15:00 | 2049.00 | 2025-09-26 10:15:00 | 1946.17 | PARTIAL | 0.50 | 5.02% |
| SELL | retest2 | 2025-09-25 10:45:00 | 2048.60 | 2025-09-26 11:15:00 | 1925.17 | PARTIAL | 0.50 | 6.02% |
| SELL | retest2 | 2025-09-25 15:15:00 | 2029.00 | 2025-09-26 11:15:00 | 1927.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 10:15:00 | 2049.00 | 2025-10-15 09:15:00 | 1843.74 | TARGET_HIT | 0.50 | 10.02% |
| SELL | retest2 | 2025-09-25 10:45:00 | 2048.60 | 2025-10-16 12:15:00 | 1948.70 | STOP_HIT | 0.50 | 4.88% |
| SELL | retest2 | 2025-09-25 15:15:00 | 2029.00 | 2025-10-16 12:15:00 | 1948.70 | STOP_HIT | 0.50 | 3.96% |
