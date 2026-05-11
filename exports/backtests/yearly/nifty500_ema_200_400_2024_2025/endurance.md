# Endurance Technologies Ltd. (ENDURANCE)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 2530.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 39 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 49 |
| PARTIAL | 17 |
| TARGET_HIT | 19 |
| STOP_HIT | 30 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 66 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 41 / 25
- **Target hits / Stop hits / Partials:** 19 / 30 / 17
- **Avg / median % per leg:** 3.15% / 5.00%
- **Sum % (uncompounded):** 208.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 7 | 77.8% | 7 | 2 | 0 | 7.41% | 66.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 7 | 77.8% | 7 | 2 | 0 | 7.41% | 66.7% |
| SELL (all) | 57 | 34 | 59.6% | 12 | 28 | 17 | 2.48% | 141.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 57 | 34 | 59.6% | 12 | 28 | 17 | 2.48% | 141.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 66 | 41 | 62.1% | 19 | 30 | 17 | 3.15% | 208.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 12:15:00 | 2381.00 | 2480.29 | 2480.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 09:15:00 | 2366.85 | 2476.31 | 2478.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 10:15:00 | 2391.10 | 2385.86 | 2423.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-16 11:00:00 | 2391.10 | 2385.86 | 2423.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 2441.60 | 2385.80 | 2422.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 15:00:00 | 2441.60 | 2385.80 | 2422.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 2441.10 | 2386.35 | 2422.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 09:15:00 | 2383.10 | 2386.35 | 2422.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 09:45:00 | 2426.55 | 2379.17 | 2403.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 10:45:00 | 2425.05 | 2379.66 | 2403.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 11:15:00 | 2425.00 | 2379.66 | 2403.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 2418.80 | 2380.86 | 2403.88 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-07 14:15:00 | 2455.00 | 2381.60 | 2404.13 | SL hit (close>static) qty=1.00 sl=2450.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 15:15:00 | 2133.00 | 1960.55 | 1960.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 2187.80 | 1962.81 | 1961.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 15:15:00 | 2612.80 | 2614.79 | 2481.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-23 09:15:00 | 2603.70 | 2614.79 | 2481.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 2495.20 | 2598.04 | 2504.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 13:45:00 | 2499.60 | 2598.04 | 2504.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 2478.20 | 2596.85 | 2504.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 2478.20 | 2596.85 | 2504.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 2512.80 | 2590.11 | 2512.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 13:00:00 | 2512.80 | 2590.11 | 2512.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 2507.00 | 2589.28 | 2512.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 13:30:00 | 2499.90 | 2589.28 | 2512.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 2517.70 | 2588.57 | 2512.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 2530.10 | 2587.88 | 2512.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 10:45:00 | 2535.00 | 2582.61 | 2512.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 13:00:00 | 2527.30 | 2581.64 | 2512.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 2536.60 | 2579.77 | 2513.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 2538.40 | 2579.36 | 2513.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 10:15:00 | 2542.30 | 2579.36 | 2513.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 12:30:00 | 2549.90 | 2578.48 | 2513.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 15:15:00 | 2505.00 | 2576.94 | 2513.91 | SL hit (close<static) qty=1.00 sl=2511.30 alert=retest2 |

### Cycle 3 — SELL (started 2025-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 11:15:00 | 2677.40 | 2797.01 | 2797.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 13:15:00 | 2648.00 | 2794.29 | 2796.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 2696.80 | 2670.22 | 2715.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 15:00:00 | 2696.80 | 2670.22 | 2715.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 2667.70 | 2670.49 | 2715.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 10:30:00 | 2656.70 | 2670.34 | 2714.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 12:45:00 | 2653.30 | 2670.22 | 2714.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 14:00:00 | 2660.00 | 2670.12 | 2713.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 14:30:00 | 2661.60 | 2670.01 | 2713.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-13 09:15:00 | 2523.86 | 2611.77 | 2660.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-13 09:15:00 | 2527.00 | 2611.77 | 2660.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-13 09:15:00 | 2528.52 | 2611.77 | 2660.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 14:15:00 | 2520.64 | 2605.50 | 2654.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-22 15:15:00 | 2391.03 | 2566.26 | 2625.10 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-10-17 09:15:00 | 2383.10 | 2024-11-07 14:15:00 | 2455.00 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2024-11-07 09:45:00 | 2426.55 | 2024-11-07 14:15:00 | 2455.00 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-11-07 10:45:00 | 2425.05 | 2024-11-07 14:15:00 | 2455.00 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2024-11-07 11:15:00 | 2425.00 | 2024-11-07 14:15:00 | 2455.00 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2024-11-07 14:30:00 | 2413.00 | 2024-11-11 09:15:00 | 2519.30 | STOP_HIT | 1.00 | -4.41% |
| SELL | retest2 | 2024-11-08 09:45:00 | 2416.45 | 2024-11-11 09:15:00 | 2519.30 | STOP_HIT | 1.00 | -4.26% |
| SELL | retest2 | 2024-11-08 10:15:00 | 2414.00 | 2024-11-11 09:15:00 | 2519.30 | STOP_HIT | 1.00 | -4.36% |
| SELL | retest2 | 2024-11-08 13:00:00 | 2413.00 | 2024-11-11 09:15:00 | 2519.30 | STOP_HIT | 1.00 | -4.41% |
| SELL | retest2 | 2024-11-14 12:30:00 | 2400.15 | 2024-11-25 09:15:00 | 2488.10 | STOP_HIT | 1.00 | -3.66% |
| SELL | retest2 | 2024-11-25 11:30:00 | 2401.00 | 2024-12-13 09:15:00 | 2280.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-25 12:00:00 | 2400.15 | 2024-12-13 09:15:00 | 2280.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-25 12:45:00 | 2386.05 | 2024-12-13 09:15:00 | 2266.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-28 10:45:00 | 2363.75 | 2024-12-17 13:15:00 | 2245.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-02 09:45:00 | 2363.15 | 2024-12-17 13:15:00 | 2244.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-02 12:15:00 | 2363.50 | 2024-12-17 13:15:00 | 2245.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-03 09:45:00 | 2359.95 | 2024-12-17 13:15:00 | 2241.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-25 11:30:00 | 2401.00 | 2024-12-23 11:15:00 | 2160.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-25 12:00:00 | 2400.15 | 2024-12-23 12:15:00 | 2160.14 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-25 12:45:00 | 2386.05 | 2024-12-23 14:15:00 | 2147.45 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-28 10:45:00 | 2363.75 | 2024-12-30 14:15:00 | 2127.38 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-02 09:45:00 | 2363.15 | 2024-12-30 14:15:00 | 2126.84 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-02 12:15:00 | 2363.50 | 2024-12-30 14:15:00 | 2127.15 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-03 09:45:00 | 2359.95 | 2024-12-30 14:15:00 | 2123.95 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-18 12:30:00 | 1947.60 | 2025-03-24 09:15:00 | 1993.05 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-03-20 12:45:00 | 1948.00 | 2025-03-24 09:15:00 | 1993.05 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-03-20 13:45:00 | 1949.45 | 2025-03-24 09:15:00 | 1993.05 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-03-21 11:30:00 | 1945.25 | 2025-03-24 09:15:00 | 1993.05 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-03-27 11:30:00 | 1973.85 | 2025-04-01 09:15:00 | 2007.70 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-04-01 11:15:00 | 1962.85 | 2025-04-04 09:15:00 | 1864.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 11:15:00 | 1962.85 | 2025-04-07 09:15:00 | 1766.57 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-17 12:30:00 | 1974.50 | 2025-04-17 14:15:00 | 1875.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-17 12:30:00 | 1974.50 | 2025-04-17 14:15:00 | 1912.30 | STOP_HIT | 0.50 | 3.15% |
| SELL | retest2 | 2025-04-21 13:30:00 | 1950.50 | 2025-05-02 12:15:00 | 1852.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-25 09:15:00 | 1941.50 | 2025-05-02 12:15:00 | 1844.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-21 13:30:00 | 1950.50 | 2025-05-07 09:15:00 | 1922.70 | STOP_HIT | 0.50 | 1.43% |
| SELL | retest2 | 2025-04-25 09:15:00 | 1941.50 | 2025-05-07 09:15:00 | 1922.70 | STOP_HIT | 0.50 | 0.97% |
| BUY | retest2 | 2025-08-08 09:15:00 | 2530.10 | 2025-08-12 15:15:00 | 2505.00 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-08-11 10:45:00 | 2535.00 | 2025-08-12 15:15:00 | 2505.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-08-11 13:00:00 | 2527.30 | 2025-08-18 09:15:00 | 2783.11 | TARGET_HIT | 1.00 | 10.12% |
| BUY | retest2 | 2025-08-12 09:15:00 | 2536.60 | 2025-08-18 09:15:00 | 2788.50 | TARGET_HIT | 1.00 | 9.93% |
| BUY | retest2 | 2025-08-12 10:15:00 | 2542.30 | 2025-08-18 09:15:00 | 2780.03 | TARGET_HIT | 1.00 | 9.35% |
| BUY | retest2 | 2025-08-12 12:30:00 | 2549.90 | 2025-08-18 09:15:00 | 2790.26 | TARGET_HIT | 1.00 | 9.43% |
| BUY | retest2 | 2025-08-13 09:15:00 | 2544.30 | 2025-08-18 10:15:00 | 2798.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-13 09:45:00 | 2546.60 | 2025-08-18 10:15:00 | 2801.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-14 09:15:00 | 2598.00 | 2025-08-18 10:15:00 | 2857.80 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-12-22 10:30:00 | 2656.70 | 2026-01-13 09:15:00 | 2523.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-22 12:45:00 | 2653.30 | 2026-01-13 09:15:00 | 2527.00 | PARTIAL | 0.50 | 4.76% |
| SELL | retest2 | 2025-12-22 14:00:00 | 2660.00 | 2026-01-13 09:15:00 | 2528.52 | PARTIAL | 0.50 | 4.94% |
| SELL | retest2 | 2025-12-22 14:30:00 | 2661.60 | 2026-01-14 14:15:00 | 2520.64 | PARTIAL | 0.50 | 5.30% |
| SELL | retest2 | 2025-12-22 10:30:00 | 2656.70 | 2026-01-22 15:15:00 | 2391.03 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-22 12:45:00 | 2653.30 | 2026-01-22 15:15:00 | 2387.97 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-22 14:00:00 | 2660.00 | 2026-01-22 15:15:00 | 2394.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-22 14:30:00 | 2661.60 | 2026-01-22 15:15:00 | 2395.44 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-03 11:00:00 | 2481.60 | 2026-02-11 10:15:00 | 2603.80 | STOP_HIT | 1.00 | -4.92% |
| SELL | retest2 | 2026-02-03 12:30:00 | 2484.80 | 2026-02-11 10:15:00 | 2603.80 | STOP_HIT | 1.00 | -4.79% |
| SELL | retest2 | 2026-02-04 10:30:00 | 2487.00 | 2026-02-11 10:15:00 | 2603.80 | STOP_HIT | 1.00 | -4.70% |
| SELL | retest2 | 2026-02-09 11:45:00 | 2477.40 | 2026-02-11 10:15:00 | 2603.80 | STOP_HIT | 1.00 | -5.10% |
| SELL | retest2 | 2026-02-12 13:30:00 | 2534.20 | 2026-02-24 12:15:00 | 2620.40 | STOP_HIT | 1.00 | -3.40% |
| SELL | retest2 | 2026-02-12 14:15:00 | 2546.90 | 2026-02-24 12:15:00 | 2620.40 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2026-02-23 14:30:00 | 2554.50 | 2026-02-24 12:15:00 | 2620.40 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2026-03-04 09:15:00 | 2559.30 | 2026-03-09 09:15:00 | 2431.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-04 09:15:00 | 2559.30 | 2026-03-11 11:15:00 | 2532.30 | STOP_HIT | 0.50 | 1.05% |
| SELL | retest2 | 2026-04-13 09:15:00 | 2396.90 | 2026-04-23 09:15:00 | 2462.90 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2026-04-22 12:30:00 | 2414.20 | 2026-04-23 09:15:00 | 2462.90 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2026-04-23 10:15:00 | 2422.50 | 2026-04-30 11:15:00 | 2301.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-23 10:15:00 | 2422.50 | 2026-05-04 10:15:00 | 2378.80 | STOP_HIT | 0.50 | 1.80% |
