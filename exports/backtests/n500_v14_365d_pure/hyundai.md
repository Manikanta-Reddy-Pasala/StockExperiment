# Hyundai Motor India Ltd. (HYUNDAI)

## Backtest Summary

- **Window:** 2025-01-15 09:15:00 → 2026-05-08 15:15:00 (2263 bars)
- **Last close:** 1833.10
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
| ALERT3 | 19 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 11 |
| PARTIAL | 10 |
| TARGET_HIT | 2 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 1
- **Target hits / Stop hits / Partials:** 2 / 9 / 10
- **Avg / median % per leg:** 4.37% / 5.00%
- **Sum % (uncompounded):** 91.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.80% | -3.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.80% | -3.8% |
| SELL (all) | 20 | 20 | 100.0% | 2 | 8 | 10 | 4.77% | 95.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 20 | 20 | 100.0% | 2 | 8 | 10 | 4.77% | 95.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 21 | 20 | 95.2% | 2 | 9 | 10 | 4.37% | 91.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 13:15:00 | 1821.00 | 1724.87 | 1724.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 14:15:00 | 1870.40 | 1726.31 | 1725.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 13:15:00 | 2059.60 | 2073.14 | 1997.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-28 14:00:00 | 2059.60 | 2073.14 | 1997.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 2419.30 | 2539.10 | 2423.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:00:00 | 2419.30 | 2539.10 | 2423.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 2398.20 | 2537.70 | 2423.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:00:00 | 2398.20 | 2537.70 | 2423.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 2408.90 | 2535.19 | 2423.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:45:00 | 2405.60 | 2535.19 | 2423.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 2407.30 | 2532.83 | 2423.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:45:00 | 2406.60 | 2532.83 | 2423.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 2409.50 | 2531.60 | 2423.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 2402.50 | 2531.60 | 2423.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 2390.20 | 2511.71 | 2421.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 2390.20 | 2511.71 | 2421.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 2469.20 | 2504.78 | 2420.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 11:15:00 | 2470.50 | 2504.78 | 2420.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 10:15:00 | 2376.50 | 2499.67 | 2421.00 | SL hit (close<static) qty=1.00 sl=2406.80 alert=retest2 |

### Cycle 2 — SELL (started 2025-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 10:15:00 | 2322.00 | 2384.34 | 2384.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 11:15:00 | 2314.50 | 2383.65 | 2384.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 10:15:00 | 2373.60 | 2367.19 | 2375.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 10:15:00 | 2373.60 | 2367.19 | 2375.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 2373.60 | 2367.19 | 2375.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:00:00 | 2373.60 | 2367.19 | 2375.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 2388.50 | 2367.40 | 2375.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:30:00 | 2387.40 | 2367.40 | 2375.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 2399.20 | 2367.72 | 2375.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:00:00 | 2399.20 | 2367.72 | 2375.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 2386.70 | 2368.48 | 2375.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:00:00 | 2386.70 | 2368.48 | 2375.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 2389.00 | 2368.68 | 2375.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:30:00 | 2384.10 | 2368.68 | 2375.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 2341.00 | 2368.50 | 2375.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 11:15:00 | 2323.70 | 2369.15 | 2375.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 14:45:00 | 2334.90 | 2349.57 | 2363.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 09:15:00 | 2319.30 | 2349.49 | 2363.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 09:15:00 | 2284.20 | 2347.94 | 2362.39 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 2342.00 | 2313.05 | 2333.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 2342.00 | 2313.05 | 2333.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 2334.30 | 2313.26 | 2333.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:15:00 | 2342.50 | 2313.26 | 2333.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 2335.40 | 2313.48 | 2333.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 09:15:00 | 2312.00 | 2316.91 | 2334.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:30:00 | 2314.70 | 2310.92 | 2328.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 2291.10 | 2312.79 | 2328.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 13:45:00 | 2325.00 | 2312.96 | 2328.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 2316.00 | 2312.99 | 2328.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 14:45:00 | 2328.60 | 2312.99 | 2328.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 2207.51 | 2302.88 | 2321.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 2218.16 | 2302.88 | 2321.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 2203.34 | 2302.88 | 2321.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 2169.99 | 2302.88 | 2321.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 2196.40 | 2302.88 | 2321.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 2198.96 | 2302.88 | 2321.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 2176.54 | 2302.88 | 2321.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 2208.75 | 2302.88 | 2321.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 14:15:00 | 2239.30 | 2212.99 | 2253.95 | SL hit (close>ema200) qty=0.50 sl=2212.99 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 14:15:00 | 2239.30 | 2212.99 | 2253.95 | SL hit (close>ema200) qty=0.50 sl=2212.99 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 14:15:00 | 2239.30 | 2212.99 | 2253.95 | SL hit (close>ema200) qty=0.50 sl=2212.99 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 14:15:00 | 2239.30 | 2212.99 | 2253.95 | SL hit (close>ema200) qty=0.50 sl=2212.99 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 14:15:00 | 2239.30 | 2212.99 | 2253.95 | SL hit (close>ema200) qty=0.50 sl=2212.99 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 14:15:00 | 2239.30 | 2212.99 | 2253.95 | SL hit (close>ema200) qty=0.50 sl=2212.99 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 14:15:00 | 2239.30 | 2212.99 | 2253.95 | SL hit (close>ema200) qty=0.50 sl=2212.99 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 14:15:00 | 2239.30 | 2212.99 | 2253.95 | SL hit (close>ema200) qty=0.50 sl=2212.99 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 2325.40 | 2211.78 | 2250.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 15:00:00 | 2325.40 | 2211.78 | 2250.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 2250.00 | 2212.16 | 2250.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:00:00 | 2242.10 | 2212.45 | 2250.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 2244.40 | 2215.70 | 2251.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2129.99 | 2206.26 | 2241.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2132.18 | 2206.26 | 2241.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 2017.89 | 2180.58 | 2223.00 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 2019.96 | 2180.58 | 2223.00 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-10-15 11:15:00 | 2470.50 | 2025-10-16 10:15:00 | 2376.50 | STOP_HIT | 1.00 | -3.80% |
| SELL | retest2 | 2025-12-05 11:15:00 | 2323.70 | 2026-01-27 09:15:00 | 2207.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-11 14:45:00 | 2334.90 | 2026-01-27 09:15:00 | 2218.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 09:15:00 | 2319.30 | 2026-01-27 09:15:00 | 2203.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-15 09:15:00 | 2284.20 | 2026-01-27 09:15:00 | 2169.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 09:15:00 | 2312.00 | 2026-01-27 09:15:00 | 2196.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 09:30:00 | 2314.70 | 2026-01-27 09:15:00 | 2198.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 09:15:00 | 2291.10 | 2026-01-27 09:15:00 | 2176.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 13:45:00 | 2325.00 | 2026-01-27 09:15:00 | 2208.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 11:15:00 | 2323.70 | 2026-02-18 14:15:00 | 2239.30 | STOP_HIT | 0.50 | 3.63% |
| SELL | retest2 | 2025-12-11 14:45:00 | 2334.90 | 2026-02-18 14:15:00 | 2239.30 | STOP_HIT | 0.50 | 4.09% |
| SELL | retest2 | 2025-12-12 09:15:00 | 2319.30 | 2026-02-18 14:15:00 | 2239.30 | STOP_HIT | 0.50 | 3.45% |
| SELL | retest2 | 2025-12-15 09:15:00 | 2284.20 | 2026-02-18 14:15:00 | 2239.30 | STOP_HIT | 0.50 | 1.97% |
| SELL | retest2 | 2026-01-09 09:15:00 | 2312.00 | 2026-02-18 14:15:00 | 2239.30 | STOP_HIT | 0.50 | 3.14% |
| SELL | retest2 | 2026-01-19 09:30:00 | 2314.70 | 2026-02-18 14:15:00 | 2239.30 | STOP_HIT | 0.50 | 3.26% |
| SELL | retest2 | 2026-01-20 09:15:00 | 2291.10 | 2026-02-18 14:15:00 | 2239.30 | STOP_HIT | 0.50 | 2.26% |
| SELL | retest2 | 2026-01-20 13:45:00 | 2325.00 | 2026-02-18 14:15:00 | 2239.30 | STOP_HIT | 0.50 | 3.69% |
| SELL | retest2 | 2026-02-23 10:00:00 | 2242.10 | 2026-03-02 09:15:00 | 2129.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 2244.40 | 2026-03-02 09:15:00 | 2132.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 10:00:00 | 2242.10 | 2026-03-09 09:15:00 | 2017.89 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 2244.40 | 2026-03-09 09:15:00 | 2019.96 | TARGET_HIT | 0.50 | 10.00% |
