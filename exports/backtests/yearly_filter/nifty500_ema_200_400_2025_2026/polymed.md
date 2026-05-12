# Poly Medicure Ltd. (POLYMED)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
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
| ALERT3 | 15 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 24 |
| PARTIAL | 12 |
| TARGET_HIT | 5 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 23 / 13
- **Target hits / Stop hits / Partials:** 5 / 19 / 12
- **Avg / median % per leg:** 2.60% / 4.37%
- **Sum % (uncompounded):** 93.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.27% | -15.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.27% | -15.9% |
| SELL (all) | 29 | 23 | 79.3% | 5 | 12 | 12 | 3.78% | 109.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 29 | 23 | 79.3% | 5 | 12 | 12 | 3.78% | 109.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 36 | 23 | 63.9% | 5 | 19 | 12 | 2.60% | 93.8% |

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
| SELL | retest2 | 2025-11-11 12:30:00 | 2028.10 | 2025-11-11 13:15:00 | 2070.00 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-11-12 11:00:00 | 2015.00 | 2025-11-18 09:15:00 | 1914.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 13:45:00 | 2023.00 | 2025-11-18 09:15:00 | 1921.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 11:00:00 | 2015.00 | 2025-11-24 09:15:00 | 1994.00 | STOP_HIT | 0.50 | 1.04% |
| SELL | retest2 | 2025-11-12 13:45:00 | 2023.00 | 2025-11-24 09:15:00 | 1994.00 | STOP_HIT | 0.50 | 1.43% |
| SELL | retest2 | 2025-11-14 13:30:00 | 1944.60 | 2025-11-24 09:15:00 | 1994.00 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2025-11-14 14:00:00 | 1943.50 | 2025-11-24 09:15:00 | 1994.00 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-11-14 14:45:00 | 1943.90 | 2025-11-24 09:15:00 | 1994.00 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2025-11-17 10:15:00 | 1940.50 | 2025-11-24 09:15:00 | 1994.00 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2025-11-24 13:45:00 | 1978.00 | 2025-11-24 15:15:00 | 1879.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-24 13:45:00 | 1978.00 | 2025-11-27 10:15:00 | 1933.90 | STOP_HIT | 0.50 | 2.23% |
