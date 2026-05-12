# Endurance Technologies Ltd. (ENDURANCE)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 2530.00
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
| ALERT2_SKIP | 0 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 24 |
| PARTIAL | 6 |
| TARGET_HIT | 11 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 30 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 11
- **Target hits / Stop hits / Partials:** 11 / 13 / 6
- **Avg / median % per leg:** 3.55% / 5.00%
- **Sum % (uncompounded):** 106.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 7 | 77.8% | 7 | 2 | 0 | 7.41% | 66.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 7 | 77.8% | 7 | 2 | 0 | 7.41% | 66.7% |
| SELL (all) | 21 | 12 | 57.1% | 4 | 11 | 6 | 1.89% | 39.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 21 | 12 | 57.1% | 4 | 11 | 6 | 1.89% | 39.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 30 | 19 | 63.3% | 11 | 13 | 6 | 3.55% | 106.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 15:15:00 | 2133.00 | 1960.55 | 1959.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 2187.80 | 1962.81 | 1961.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 15:15:00 | 2612.80 | 2614.79 | 2481.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-23 09:15:00 | 2603.70 | 2614.79 | 2481.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 2495.20 | 2598.04 | 2504.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 13:45:00 | 2499.60 | 2598.04 | 2504.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 2478.20 | 2596.85 | 2504.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 2478.20 | 2596.85 | 2504.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 2512.80 | 2590.11 | 2512.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 13:00:00 | 2512.80 | 2590.11 | 2512.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 2507.00 | 2589.28 | 2512.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 13:30:00 | 2499.90 | 2589.28 | 2512.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 2517.70 | 2588.57 | 2512.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 2530.10 | 2587.88 | 2512.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 10:45:00 | 2535.00 | 2582.61 | 2512.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 13:00:00 | 2527.30 | 2581.64 | 2512.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 2536.60 | 2579.77 | 2513.05 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 2538.40 | 2579.36 | 2513.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 10:15:00 | 2542.30 | 2579.36 | 2513.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 12:30:00 | 2549.90 | 2578.48 | 2513.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 15:15:00 | 2505.00 | 2576.94 | 2513.91 | SL hit (close<static) qty=1.00 sl=2511.30 alert=retest2 |

### Cycle 2 — SELL (started 2025-11-24 11:15:00)

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
