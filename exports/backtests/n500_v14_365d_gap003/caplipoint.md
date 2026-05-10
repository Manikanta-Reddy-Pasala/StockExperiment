# Caplin Point Laboratories Ltd. (CAPLIPOINT)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1854.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 14 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 11 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 11
- **Target hits / Stop hits / Partials:** 0 / 11 / 0
- **Avg / median % per leg:** -2.96% / -2.76%
- **Sum % (uncompounded):** -32.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 0 | 0.0% | 0 | 11 | 0 | -2.96% | -32.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 0 | 0.0% | 0 | 11 | 0 | -2.96% | -32.5% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 11 | 0 | 0.0% | 0 | 11 | 0 | -2.96% | -32.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 09:15:00 | 2271.90 | 1978.94 | 1977.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 14:15:00 | 2296.40 | 1993.77 | 1985.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 09:15:00 | 2055.70 | 2081.69 | 2041.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 10:00:00 | 2055.70 | 2081.69 | 2041.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 2025.10 | 2079.78 | 2041.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:00:00 | 2025.10 | 2079.78 | 2041.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 2022.00 | 2079.20 | 2041.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:00:00 | 2022.00 | 2079.20 | 2041.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 2099.70 | 2085.91 | 2052.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 11:30:00 | 2102.00 | 2086.00 | 2052.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 2043.90 | 2083.42 | 2053.35 | SL hit (close<static) qty=1.00 sl=2045.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:30:00 | 2101.70 | 2076.89 | 2052.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 11:00:00 | 2105.90 | 2077.18 | 2053.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 11:30:00 | 2101.60 | 2077.44 | 2053.39 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 1999.90 | 2083.87 | 2060.36 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-01 14:15:00 | 1999.90 | 2083.87 | 2060.36 | SL hit (close<static) qty=1.00 sl=2045.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-01 14:15:00 | 1999.90 | 2083.87 | 2060.36 | SL hit (close<static) qty=1.00 sl=2045.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-01 14:15:00 | 1999.90 | 2083.87 | 2060.36 | SL hit (close<static) qty=1.00 sl=2045.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-07-01 14:30:00 | 1999.40 | 2083.87 | 2060.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 1995.00 | 2082.98 | 2060.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 1956.10 | 2082.98 | 2060.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 2048.00 | 2064.09 | 2052.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:15:00 | 2048.60 | 2064.09 | 2052.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 2019.60 | 2063.65 | 2052.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 2019.60 | 2063.65 | 2052.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 2010.00 | 2063.12 | 2052.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:00:00 | 2010.00 | 2063.12 | 2052.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 2024.20 | 2056.11 | 2049.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 2024.20 | 2056.11 | 2049.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 2032.40 | 2053.24 | 2048.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:00:00 | 2032.40 | 2053.24 | 2048.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 2046.80 | 2052.73 | 2048.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:45:00 | 2047.90 | 2052.73 | 2048.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 2046.90 | 2052.67 | 2048.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:45:00 | 2043.60 | 2052.67 | 2048.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 2083.60 | 2052.98 | 2048.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 10:00:00 | 2095.20 | 2054.07 | 2049.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 2098.00 | 2055.25 | 2050.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 11:00:00 | 2095.50 | 2059.62 | 2052.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:15:00 | 2103.70 | 2062.11 | 2053.96 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 2054.10 | 2069.55 | 2059.18 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-25 14:15:00 | 2038.00 | 2069.03 | 2059.02 | SL hit (close<static) qty=1.00 sl=2045.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 14:15:00 | 2038.00 | 2069.03 | 2059.02 | SL hit (close<static) qty=1.00 sl=2045.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 14:15:00 | 2038.00 | 2069.03 | 2059.02 | SL hit (close<static) qty=1.00 sl=2045.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 14:15:00 | 2038.00 | 2069.03 | 2059.02 | SL hit (close<static) qty=1.00 sl=2045.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 09:30:00 | 2065.40 | 2068.77 | 2058.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 14:15:00 | 2046.50 | 2068.44 | 2059.07 | SL hit (close<static) qty=1.00 sl=2050.60 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 14:30:00 | 2064.70 | 2068.44 | 2059.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 10:15:00 | 2041.00 | 2067.96 | 2058.97 | SL hit (close<static) qty=1.00 sl=2050.60 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 12:45:00 | 2071.20 | 2067.88 | 2059.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 2039.60 | 2073.36 | 2062.66 | SL hit (close<static) qty=1.00 sl=2050.60 alert=retest2 |
| CROSSOVER_SKIP | 2025-08-06 12:15:00 | 1930.00 | 2053.01 | 2053.21 | min_gap filter: gap=0.010% < 0.030% |
| TREND_RESET | 2025-08-06 12:15:00 | 1930.00 | 2053.01 | 2053.21 | EMA inversion without crossover edge (EMA200=2053.01 EMA400=2053.21) — end cycle |
| CROSSOVER_SKIP | 2025-08-12 11:15:00 | 2139.50 | 2053.21 | 2052.95 | min_gap filter: gap=0.012% < 0.030% |
| CROSSOVER_SKIP | 2025-10-10 09:15:00 | 2038.50 | 2120.31 | 2120.71 | min_gap filter: gap=0.020% < 0.030% |

### Cycle 2 — BUY (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 09:15:00 | 1860.70 | 1725.35 | 1724.68 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-18 11:30:00 | 2102.00 | 2025-06-20 09:15:00 | 2043.90 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2025-06-25 09:30:00 | 2101.70 | 2025-07-01 14:15:00 | 1999.90 | STOP_HIT | 1.00 | -4.84% |
| BUY | retest2 | 2025-06-25 11:00:00 | 2105.90 | 2025-07-01 14:15:00 | 1999.90 | STOP_HIT | 1.00 | -5.03% |
| BUY | retest2 | 2025-06-25 11:30:00 | 2101.60 | 2025-07-01 14:15:00 | 1999.90 | STOP_HIT | 1.00 | -4.84% |
| BUY | retest2 | 2025-07-16 10:00:00 | 2095.20 | 2025-07-25 14:15:00 | 2038.00 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2025-07-17 09:15:00 | 2098.00 | 2025-07-25 14:15:00 | 2038.00 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2025-07-18 11:00:00 | 2095.50 | 2025-07-25 14:15:00 | 2038.00 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2025-07-21 10:15:00 | 2103.70 | 2025-07-25 14:15:00 | 2038.00 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2025-07-28 09:30:00 | 2065.40 | 2025-07-28 14:15:00 | 2046.50 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-07-28 14:30:00 | 2064.70 | 2025-07-29 10:15:00 | 2041.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-07-29 12:45:00 | 2071.20 | 2025-08-01 09:15:00 | 2039.60 | STOP_HIT | 1.00 | -1.53% |
