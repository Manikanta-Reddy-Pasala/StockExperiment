# Poly Medicure Ltd. (POLYMED)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1649.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 4 |
| ALERT3 | 27 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 39 |
| PARTIAL | 16 |
| TARGET_HIT | 16 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 55 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 41 / 14
- **Target hits / Stop hits / Partials:** 16 / 23 / 16
- **Avg / median % per leg:** 4.12% / 5.00%
- **Sum % (uncompounded):** 226.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 10 | 55.6% | 10 | 8 | 0 | 4.52% | 81.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 18 | 10 | 55.6% | 10 | 8 | 0 | 4.52% | 81.4% |
| SELL (all) | 37 | 31 | 83.8% | 6 | 15 | 16 | 3.92% | 145.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 37 | 31 | 83.8% | 6 | 15 | 16 | 3.92% | 145.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 55 | 41 | 74.5% | 16 | 23 | 16 | 4.12% | 226.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-29 11:15:00 | 1393.25 | 1467.91 | 1468.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-30 09:15:00 | 1386.80 | 1464.25 | 1466.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-01 09:15:00 | 1481.25 | 1458.01 | 1462.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 09:15:00 | 1481.25 | 1458.01 | 1462.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 09:15:00 | 1481.25 | 1458.01 | 1462.79 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 09:15:00 | 1534.75 | 1467.17 | 1467.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-06 11:15:00 | 1573.30 | 1468.98 | 1468.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 09:15:00 | 1558.00 | 1560.58 | 1530.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-11 14:15:00 | 1527.00 | 1559.45 | 1530.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 14:15:00 | 1527.00 | 1559.45 | 1530.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 1562.60 | 1546.39 | 1532.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-15 09:15:00 | 1520.50 | 1546.46 | 1533.10 | SL hit (close<static) qty=1.00 sl=1523.05 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 14:15:00 | 2407.40 | 2623.04 | 2623.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-23 09:15:00 | 2348.30 | 2618.07 | 2621.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 11:15:00 | 2530.00 | 2483.84 | 2542.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 11:15:00 | 2530.00 | 2483.84 | 2542.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 11:15:00 | 2530.00 | 2483.84 | 2542.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 12:00:00 | 2530.00 | 2483.84 | 2542.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 2342.55 | 2249.80 | 2358.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:45:00 | 2342.80 | 2249.80 | 2358.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 2316.85 | 2244.49 | 2323.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 15:00:00 | 2316.85 | 2244.49 | 2323.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 2288.00 | 2244.93 | 2323.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 09:15:00 | 2329.75 | 2244.93 | 2323.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 2337.90 | 2245.85 | 2323.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 09:45:00 | 2348.20 | 2245.85 | 2323.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 10:15:00 | 2299.35 | 2246.38 | 2323.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-24 12:00:00 | 2296.60 | 2246.88 | 2323.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-24 12:45:00 | 2296.90 | 2247.47 | 2323.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-24 13:30:00 | 2292.05 | 2247.85 | 2323.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 09:15:00 | 2254.95 | 2248.88 | 2322.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-26 09:15:00 | 2181.77 | 2247.02 | 2319.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-26 09:15:00 | 2182.05 | 2247.02 | 2319.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-26 09:15:00 | 2177.45 | 2247.02 | 2319.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-28 11:15:00 | 2254.00 | 2241.64 | 2310.72 | SL hit (close>ema200) qty=0.50 sl=2241.64 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 10:15:00 | 2585.30 | 2311.44 | 2311.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 15:15:00 | 2602.20 | 2366.39 | 2340.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 09:15:00 | 2450.30 | 2455.44 | 2394.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 2403.10 | 2453.19 | 2395.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 2403.10 | 2453.19 | 2395.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 13:45:00 | 2439.90 | 2436.27 | 2393.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 14:15:00 | 2438.40 | 2436.27 | 2393.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 14:30:00 | 2439.00 | 2437.84 | 2395.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 11:00:00 | 2452.90 | 2442.67 | 2401.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 2408.90 | 2442.08 | 2401.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 13:45:00 | 2406.20 | 2442.08 | 2401.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 2409.00 | 2441.38 | 2401.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:15:00 | 2384.30 | 2441.38 | 2401.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 2388.50 | 2440.85 | 2401.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:30:00 | 2373.20 | 2440.85 | 2401.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 2407.40 | 2440.52 | 2401.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 11:45:00 | 2413.20 | 2440.27 | 2402.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 13:45:00 | 2410.00 | 2439.68 | 2402.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 14:15:00 | 2422.60 | 2439.68 | 2402.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 11:15:00 | 2379.80 | 2438.02 | 2402.17 | SL hit (close<static) qty=1.00 sl=2385.20 alert=retest2 |

### Cycle 5 — SELL (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 10:15:00 | 2243.60 | 2374.75 | 2375.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 12:15:00 | 2235.10 | 2372.09 | 2373.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 14:15:00 | 2233.10 | 2231.85 | 2284.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-30 15:00:00 | 2233.10 | 2231.85 | 2284.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 2251.40 | 2233.45 | 2283.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 13:15:00 | 2237.40 | 2234.24 | 2282.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 09:30:00 | 2240.80 | 2235.30 | 2282.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:30:00 | 2244.50 | 2235.56 | 2281.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 13:15:00 | 2242.20 | 2235.56 | 2281.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-16 11:15:00 | 2128.76 | 2224.05 | 2263.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-16 11:15:00 | 2132.28 | 2224.05 | 2263.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-16 11:15:00 | 2130.09 | 2224.05 | 2263.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-16 12:15:00 | 2125.53 | 2222.99 | 2262.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-07-28 09:15:00 | 2013.66 | 2164.93 | 2220.95 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 6 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 1624.00 | 1447.03 | 1446.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 12:15:00 | 1638.80 | 1448.94 | 1447.26 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-12 09:15:00 | 1562.60 | 2024-04-15 09:15:00 | 1520.50 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2024-04-15 10:45:00 | 1544.85 | 2024-05-15 12:15:00 | 1699.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-15 11:30:00 | 1544.85 | 2024-05-15 12:15:00 | 1699.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-15 15:00:00 | 1548.85 | 2024-05-15 12:15:00 | 1703.74 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-16 10:30:00 | 1546.60 | 2024-05-15 12:15:00 | 1701.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-16 11:00:00 | 1545.95 | 2024-05-15 12:15:00 | 1700.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-24 09:30:00 | 1547.05 | 2024-05-15 12:15:00 | 1701.76 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-24 10:00:00 | 1551.80 | 2024-05-15 12:15:00 | 1706.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-04 13:30:00 | 1725.50 | 2024-06-13 09:15:00 | 1898.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-04 15:15:00 | 1720.15 | 2024-06-13 09:15:00 | 1892.17 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-05 10:30:00 | 1724.95 | 2024-06-13 09:15:00 | 1897.45 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-24 12:00:00 | 2296.60 | 2025-03-26 09:15:00 | 2181.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-24 12:45:00 | 2296.90 | 2025-03-26 09:15:00 | 2182.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-24 13:30:00 | 2292.05 | 2025-03-26 09:15:00 | 2177.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-24 12:00:00 | 2296.60 | 2025-03-28 11:15:00 | 2254.00 | STOP_HIT | 0.50 | 1.85% |
| SELL | retest2 | 2025-03-24 12:45:00 | 2296.90 | 2025-03-28 11:15:00 | 2254.00 | STOP_HIT | 0.50 | 1.87% |
| SELL | retest2 | 2025-03-24 13:30:00 | 2292.05 | 2025-03-28 11:15:00 | 2254.00 | STOP_HIT | 0.50 | 1.66% |
| SELL | retest2 | 2025-03-25 09:15:00 | 2254.95 | 2025-04-04 12:15:00 | 2142.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-25 09:15:00 | 2254.95 | 2025-04-07 09:15:00 | 2029.45 | TARGET_HIT | 0.50 | 10.00% |
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
