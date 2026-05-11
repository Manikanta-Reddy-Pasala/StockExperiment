# Caplin Point Laboratories Ltd. (CAPLIPOINT)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1854.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 24 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 16 |
| PARTIAL | 1 |
| TARGET_HIT | 3 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 13
- **Target hits / Stop hits / Partials:** 3 / 13 / 1
- **Avg / median % per leg:** -0.01% / -1.71%
- **Sum % (uncompounded):** -0.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 2 | 14.3% | 2 | 12 | 0 | -1.02% | -14.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 2 | 14.3% | 2 | 12 | 0 | -1.02% | -14.2% |
| SELL (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 4.70% | 14.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 3 | 2 | 66.7% | 1 | 1 | 1 | 4.70% | 14.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 17 | 4 | 23.5% | 3 | 13 | 1 | -0.01% | -0.1% |

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

### Cycle 2 — SELL (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 12:15:00 | 1930.00 | 2053.01 | 2053.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 1900.90 | 2047.97 | 2050.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 2069.00 | 2044.02 | 2048.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 2069.00 | 2044.02 | 2048.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 2069.00 | 2044.02 | 2048.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 2069.00 | 2044.02 | 2048.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 2021.00 | 2043.79 | 2048.45 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 11:15:00 | 2139.50 | 2053.21 | 2052.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 12:15:00 | 2143.10 | 2058.18 | 2055.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 14:15:00 | 2086.70 | 2098.61 | 2079.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-26 15:15:00 | 2079.50 | 2098.61 | 2079.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 2079.50 | 2098.42 | 2079.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 09:30:00 | 2109.20 | 2098.63 | 2079.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 09:30:00 | 2113.70 | 2101.90 | 2082.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-17 10:15:00 | 2320.12 | 2161.36 | 2122.92 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 09:15:00 | 2038.50 | 2120.31 | 2120.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 13:15:00 | 2009.50 | 2111.51 | 2116.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 13:15:00 | 1952.60 | 1949.76 | 1996.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 14:00:00 | 1952.60 | 1949.76 | 1996.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 1977.90 | 1941.96 | 1975.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:45:00 | 1981.70 | 1941.96 | 1975.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 1967.40 | 1942.21 | 1975.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 12:30:00 | 1964.50 | 1944.75 | 1974.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 09:15:00 | 1981.90 | 1945.78 | 1974.82 | SL hit (close>static) qty=1.00 sl=1979.20 alert=retest2 |

### Cycle 5 — BUY (started 2026-05-08 09:15:00)

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
| BUY | retest2 | 2025-08-28 09:30:00 | 2109.20 | 2025-09-17 10:15:00 | 2320.12 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-01 09:30:00 | 2113.70 | 2025-09-17 10:15:00 | 2325.07 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-26 09:45:00 | 2092.90 | 2025-09-26 13:15:00 | 2057.20 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-12-24 12:30:00 | 1964.50 | 2025-12-26 09:15:00 | 1981.90 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-12-26 14:00:00 | 1947.80 | 2025-12-30 09:15:00 | 1850.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-26 14:00:00 | 1947.80 | 2025-12-30 12:15:00 | 1753.02 | TARGET_HIT | 0.50 | 10.00% |
