# Indiamart Intermesh Ltd. (INDIAMART)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 2091.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 18 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 25 |
| PARTIAL | 11 |
| TARGET_HIT | 2 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 14
- **Target hits / Stop hits / Partials:** 2 / 23 / 11
- **Avg / median % per leg:** 1.65% / 1.04%
- **Sum % (uncompounded):** 59.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.75% | -14.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.75% | -14.0% |
| SELL (all) | 28 | 22 | 78.6% | 2 | 15 | 11 | 2.62% | 73.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 28 | 22 | 78.6% | 2 | 15 | 11 | 2.62% | 73.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 36 | 22 | 61.1% | 2 | 23 | 11 | 1.65% | 59.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 14:15:00 | 2375.00 | 2531.32 | 2531.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 15:15:00 | 2360.00 | 2529.61 | 2530.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 10:15:00 | 2439.00 | 2411.09 | 2452.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 10:15:00 | 2439.00 | 2411.09 | 2452.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 2439.00 | 2411.09 | 2452.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:00:00 | 2439.00 | 2411.09 | 2452.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 2461.00 | 2411.58 | 2452.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:30:00 | 2456.30 | 2411.58 | 2452.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 2439.60 | 2411.86 | 2452.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 10:30:00 | 2428.00 | 2423.96 | 2454.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 11:15:00 | 2465.00 | 2426.61 | 2454.40 | SL hit (close>static) qty=1.00 sl=2461.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 09:15:00 | 2426.40 | 2432.88 | 2455.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 2432.80 | 2433.55 | 2454.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 10:45:00 | 2432.40 | 2433.57 | 2454.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 2451.90 | 2434.00 | 2454.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 15:00:00 | 2451.90 | 2434.00 | 2454.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 2448.00 | 2434.14 | 2454.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 2437.60 | 2434.14 | 2454.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 2484.20 | 2434.71 | 2453.91 | SL hit (close>static) qty=1.00 sl=2461.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 2484.20 | 2434.71 | 2453.91 | SL hit (close>static) qty=1.00 sl=2461.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 2484.20 | 2434.71 | 2453.91 | SL hit (close>static) qty=1.00 sl=2461.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 2484.20 | 2434.71 | 2453.91 | SL hit (close>static) qty=1.00 sl=2458.10 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 09:30:00 | 2442.00 | 2442.94 | 2456.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 11:15:00 | 2466.60 | 2443.29 | 2456.43 | SL hit (close>static) qty=1.00 sl=2458.10 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 10:00:00 | 2445.00 | 2444.58 | 2456.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 11:00:00 | 2441.10 | 2444.55 | 2456.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 10:15:00 | 2322.75 | 2437.21 | 2451.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 10:15:00 | 2319.04 | 2437.21 | 2451.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-29 10:15:00 | 2200.50 | 2288.97 | 2343.67 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-12-29 11:15:00 | 2196.99 | 2288.10 | 2342.96 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 2282.80 | 2217.49 | 2281.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:30:00 | 2281.20 | 2217.49 | 2281.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 2275.00 | 2218.06 | 2281.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:30:00 | 2287.70 | 2218.06 | 2281.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 2293.50 | 2218.81 | 2281.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:45:00 | 2292.00 | 2218.81 | 2281.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 2292.10 | 2219.54 | 2281.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:15:00 | 2292.80 | 2219.54 | 2281.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 2290.50 | 2220.25 | 2281.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:15:00 | 2294.50 | 2220.25 | 2281.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 2289.30 | 2220.93 | 2281.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 15:15:00 | 2265.00 | 2220.93 | 2281.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 2223.30 | 2221.39 | 2281.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 11:45:00 | 2207.70 | 2221.19 | 2280.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 10:15:00 | 2198.50 | 2215.94 | 2274.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 11:30:00 | 2199.10 | 2216.15 | 2273.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 12:45:00 | 2203.00 | 2217.64 | 2272.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2251.80 | 2207.67 | 2253.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 12:15:00 | 2218.50 | 2219.21 | 2254.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 13:00:00 | 2212.00 | 2219.13 | 2254.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 2213.10 | 2224.83 | 2253.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:15:00 | 2205.90 | 2220.51 | 2247.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 2229.90 | 2211.42 | 2238.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:00:00 | 2211.70 | 2211.43 | 2237.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2097.31 | 2203.66 | 2231.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2088.57 | 2203.66 | 2231.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2089.14 | 2203.66 | 2231.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2092.85 | 2203.66 | 2231.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2107.57 | 2203.66 | 2231.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2101.40 | 2203.66 | 2231.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2102.44 | 2203.66 | 2231.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2095.61 | 2203.66 | 2231.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2101.11 | 2203.66 | 2231.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 14:15:00 | 2184.80 | 2174.23 | 2211.11 | SL hit (close>ema200) qty=0.50 sl=2174.23 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 14:15:00 | 2184.80 | 2174.23 | 2211.11 | SL hit (close>ema200) qty=0.50 sl=2174.23 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 14:15:00 | 2184.80 | 2174.23 | 2211.11 | SL hit (close>ema200) qty=0.50 sl=2174.23 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 14:15:00 | 2184.80 | 2174.23 | 2211.11 | SL hit (close>ema200) qty=0.50 sl=2174.23 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 14:15:00 | 2184.80 | 2174.23 | 2211.11 | SL hit (close>ema200) qty=0.50 sl=2174.23 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 14:15:00 | 2184.80 | 2174.23 | 2211.11 | SL hit (close>ema200) qty=0.50 sl=2174.23 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 14:15:00 | 2184.80 | 2174.23 | 2211.11 | SL hit (close>ema200) qty=0.50 sl=2174.23 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 14:15:00 | 2184.80 | 2174.23 | 2211.11 | SL hit (close>ema200) qty=0.50 sl=2174.23 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 14:15:00 | 2184.80 | 2174.23 | 2211.11 | SL hit (close>ema200) qty=0.50 sl=2174.23 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-08-05 12:30:00 | 2490.20 | 2025-09-19 09:15:00 | 2487.00 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-08-05 13:00:00 | 2488.80 | 2025-09-19 09:15:00 | 2487.00 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2025-08-05 14:30:00 | 2487.50 | 2025-09-19 09:15:00 | 2487.00 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2025-08-06 09:15:00 | 2545.00 | 2025-09-19 09:15:00 | 2487.00 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2025-08-07 14:00:00 | 2526.20 | 2025-09-22 09:15:00 | 2449.30 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2025-08-12 14:30:00 | 2519.50 | 2025-09-22 09:15:00 | 2449.30 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2025-08-12 15:00:00 | 2523.10 | 2025-09-22 09:15:00 | 2449.30 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2025-09-18 14:45:00 | 2519.20 | 2025-09-22 09:15:00 | 2449.30 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2025-10-31 10:30:00 | 2428.00 | 2025-11-03 11:15:00 | 2465.00 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-11-07 09:15:00 | 2426.40 | 2025-11-12 09:15:00 | 2484.20 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2025-11-10 09:15:00 | 2432.80 | 2025-11-12 09:15:00 | 2484.20 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-11-10 10:45:00 | 2432.40 | 2025-11-12 09:15:00 | 2484.20 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-11-11 09:15:00 | 2437.60 | 2025-11-12 09:15:00 | 2484.20 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-11-17 09:30:00 | 2442.00 | 2025-11-17 11:15:00 | 2466.60 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-11-18 10:00:00 | 2445.00 | 2025-11-21 10:15:00 | 2322.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 11:00:00 | 2441.10 | 2025-11-21 10:15:00 | 2319.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 10:00:00 | 2445.00 | 2025-12-29 10:15:00 | 2200.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-18 11:00:00 | 2441.10 | 2025-12-29 11:15:00 | 2196.99 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-19 11:45:00 | 2207.70 | 2026-03-02 09:15:00 | 2097.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-21 10:15:00 | 2198.50 | 2026-03-02 09:15:00 | 2088.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-21 11:30:00 | 2199.10 | 2026-03-02 09:15:00 | 2089.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 12:45:00 | 2203.00 | 2026-03-02 09:15:00 | 2092.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-06 12:15:00 | 2218.50 | 2026-03-02 09:15:00 | 2107.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-06 13:00:00 | 2212.00 | 2026-03-02 09:15:00 | 2101.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 09:15:00 | 2213.10 | 2026-03-02 09:15:00 | 2102.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 09:15:00 | 2205.90 | 2026-03-02 09:15:00 | 2095.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 11:00:00 | 2211.70 | 2026-03-02 09:15:00 | 2101.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 11:45:00 | 2207.70 | 2026-03-09 14:15:00 | 2184.80 | STOP_HIT | 0.50 | 1.04% |
| SELL | retest2 | 2026-01-21 10:15:00 | 2198.50 | 2026-03-09 14:15:00 | 2184.80 | STOP_HIT | 0.50 | 0.62% |
| SELL | retest2 | 2026-01-21 11:30:00 | 2199.10 | 2026-03-09 14:15:00 | 2184.80 | STOP_HIT | 0.50 | 0.65% |
| SELL | retest2 | 2026-01-22 12:45:00 | 2203.00 | 2026-03-09 14:15:00 | 2184.80 | STOP_HIT | 0.50 | 0.83% |
| SELL | retest2 | 2026-02-06 12:15:00 | 2218.50 | 2026-03-09 14:15:00 | 2184.80 | STOP_HIT | 0.50 | 1.52% |
| SELL | retest2 | 2026-02-06 13:00:00 | 2212.00 | 2026-03-09 14:15:00 | 2184.80 | STOP_HIT | 0.50 | 1.23% |
| SELL | retest2 | 2026-02-12 09:15:00 | 2213.10 | 2026-03-09 14:15:00 | 2184.80 | STOP_HIT | 0.50 | 1.28% |
| SELL | retest2 | 2026-02-18 09:15:00 | 2205.90 | 2026-03-09 14:15:00 | 2184.80 | STOP_HIT | 0.50 | 0.96% |
| SELL | retest2 | 2026-02-25 11:00:00 | 2211.70 | 2026-03-09 14:15:00 | 2184.80 | STOP_HIT | 0.50 | 1.22% |
