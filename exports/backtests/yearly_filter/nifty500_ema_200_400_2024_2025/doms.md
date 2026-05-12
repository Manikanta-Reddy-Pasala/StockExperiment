# DOMS Industries Ltd. (DOMS)

## Backtest Summary

- **Window:** 2023-12-20 09:15:00 → 2026-05-11 15:15:00 (4119 bars)
- **Last close:** 2320.00
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
| ALERT2_SKIP | 0 |
| ALERT3 | 38 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 45 |
| PARTIAL | 13 |
| TARGET_HIT | 3 |
| STOP_HIT | 36 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 52 (incl. partial bookings)
- **Trades open at end:** 6
- **Winners / losers:** 21 / 31
- **Target hits / Stop hits / Partials:** 3 / 36 / 13
- **Avg / median % per leg:** 0.52% / -0.89%
- **Sum % (uncompounded):** 26.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -4.79% | -38.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -4.79% | -38.4% |
| SELL (all) | 44 | 21 | 47.7% | 3 | 28 | 13 | 1.48% | 65.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 44 | 21 | 47.7% | 3 | 28 | 13 | 1.48% | 65.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 52 | 21 | 40.4% | 3 | 36 | 13 | 0.52% | 26.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 12:15:00 | 2573.40 | 2744.75 | 2745.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-15 12:15:00 | 2546.10 | 2732.88 | 2739.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 13:15:00 | 2550.70 | 2530.76 | 2614.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-01 14:00:00 | 2550.70 | 2530.76 | 2614.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 2621.05 | 2530.98 | 2610.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:45:00 | 2630.00 | 2530.98 | 2610.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 2597.50 | 2531.64 | 2610.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 11:15:00 | 2582.70 | 2531.64 | 2610.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 12:00:00 | 2577.50 | 2532.10 | 2610.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-04 14:15:00 | 2635.00 | 2534.16 | 2610.26 | SL hit (close>static) qty=1.00 sl=2623.60 alert=retest2 |

### Cycle 2 — BUY (started 2025-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 14:15:00 | 2808.90 | 2621.67 | 2621.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 10:15:00 | 2842.20 | 2627.57 | 2624.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 10:15:00 | 2739.90 | 2748.12 | 2696.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-01 10:45:00 | 2742.00 | 2748.12 | 2696.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 2790.00 | 2748.23 | 2698.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 10:15:00 | 2799.55 | 2748.23 | 2698.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 10:45:00 | 2806.35 | 2748.69 | 2698.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 12:15:00 | 2795.00 | 2749.12 | 2699.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 13:15:00 | 2806.00 | 2749.46 | 2699.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 12:15:00 | 2667.00 | 2753.57 | 2705.30 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-04 12:15:00 | 2667.00 | 2753.57 | 2705.30 | SL hit (close<static) qty=1.00 sl=2693.65 alert=retest2 |

### Cycle 3 — SELL (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 09:15:00 | 2526.80 | 2727.23 | 2727.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 13:15:00 | 2490.00 | 2718.89 | 2723.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 09:15:00 | 2486.40 | 2483.48 | 2563.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-24 10:00:00 | 2486.40 | 2483.48 | 2563.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 2446.40 | 2381.72 | 2438.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:00:00 | 2446.40 | 2381.72 | 2438.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 2517.00 | 2383.07 | 2439.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:00:00 | 2517.00 | 2383.07 | 2439.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 2545.10 | 2384.68 | 2439.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:00:00 | 2545.10 | 2384.68 | 2439.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 2513.20 | 2396.67 | 2439.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:45:00 | 2505.90 | 2396.67 | 2439.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 2500.40 | 2397.70 | 2440.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 11:45:00 | 2488.40 | 2398.59 | 2440.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 15:00:00 | 2485.00 | 2401.46 | 2441.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 09:15:00 | 2477.20 | 2417.93 | 2445.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 12:00:00 | 2495.60 | 2424.14 | 2447.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 2460.50 | 2427.58 | 2448.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 11:30:00 | 2455.00 | 2427.88 | 2448.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 14:15:00 | 2459.70 | 2428.57 | 2448.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 15:00:00 | 2458.70 | 2428.87 | 2448.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 2435.70 | 2429.24 | 2448.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 2448.30 | 2429.60 | 2448.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 12:00:00 | 2448.30 | 2429.60 | 2448.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 13:15:00 | 2451.30 | 2429.95 | 2448.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 14:00:00 | 2451.30 | 2429.95 | 2448.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 2443.70 | 2430.09 | 2448.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 09:15:00 | 2436.10 | 2430.24 | 2448.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 10:00:00 | 2436.20 | 2430.47 | 2448.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 11:00:00 | 2440.60 | 2430.58 | 2448.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 14:30:00 | 2441.50 | 2430.97 | 2447.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 2446.80 | 2431.13 | 2447.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 2451.80 | 2431.13 | 2447.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 2486.10 | 2431.67 | 2448.04 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 2486.10 | 2431.67 | 2448.04 | SL hit (close>static) qty=1.00 sl=2471.90 alert=retest2 |

### Cycle 4 — BUY (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 11:15:00 | 2661.50 | 2463.52 | 2462.78 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 11:15:00 | 2390.50 | 2544.20 | 2544.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 15:15:00 | 2351.70 | 2521.40 | 2532.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 13:15:00 | 2468.90 | 2462.08 | 2495.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-04 14:00:00 | 2468.90 | 2462.08 | 2495.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 2420.20 | 2459.73 | 2493.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 2376.70 | 2454.27 | 2484.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2257.86 | 2392.89 | 2438.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-05 11:15:00 | 2139.03 | 2366.73 | 2421.55 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-02-04 11:15:00 | 2582.70 | 2025-02-04 14:15:00 | 2635.00 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-02-04 12:00:00 | 2577.50 | 2025-02-04 14:15:00 | 2635.00 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2025-02-12 09:45:00 | 2584.60 | 2025-02-14 13:15:00 | 2455.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-12 10:30:00 | 2572.00 | 2025-02-14 13:15:00 | 2443.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 09:15:00 | 2566.10 | 2025-02-14 13:15:00 | 2437.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 10:15:00 | 2597.00 | 2025-02-14 13:15:00 | 2467.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-12 09:45:00 | 2584.60 | 2025-02-18 14:15:00 | 2599.85 | STOP_HIT | 0.50 | -0.59% |
| SELL | retest2 | 2025-02-12 10:30:00 | 2572.00 | 2025-02-18 14:15:00 | 2599.85 | STOP_HIT | 0.50 | -1.08% |
| SELL | retest2 | 2025-02-13 09:15:00 | 2566.10 | 2025-02-18 14:15:00 | 2599.85 | STOP_HIT | 0.50 | -1.32% |
| SELL | retest2 | 2025-02-13 10:15:00 | 2597.00 | 2025-02-18 14:15:00 | 2599.85 | STOP_HIT | 0.50 | -0.11% |
| SELL | retest2 | 2025-02-18 15:00:00 | 2599.85 | 2025-02-28 09:15:00 | 2469.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-19 14:45:00 | 2593.15 | 2025-02-28 09:15:00 | 2463.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-24 14:45:00 | 2533.35 | 2025-02-28 13:15:00 | 2406.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 09:15:00 | 2524.65 | 2025-02-28 13:15:00 | 2403.74 | PARTIAL | 0.50 | 4.79% |
| SELL | retest2 | 2025-02-25 09:45:00 | 2527.40 | 2025-02-28 14:15:00 | 2398.42 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2025-02-27 09:30:00 | 2530.25 | 2025-02-28 14:15:00 | 2401.03 | PARTIAL | 0.50 | 5.11% |
| SELL | retest2 | 2025-02-18 15:00:00 | 2599.85 | 2025-03-03 10:15:00 | 2339.86 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-19 14:45:00 | 2593.15 | 2025-03-03 10:15:00 | 2333.84 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-24 14:45:00 | 2533.35 | 2025-03-04 14:15:00 | 2555.95 | STOP_HIT | 0.50 | -0.89% |
| SELL | retest2 | 2025-02-25 09:15:00 | 2524.65 | 2025-03-04 14:15:00 | 2555.95 | STOP_HIT | 0.50 | -1.24% |
| SELL | retest2 | 2025-02-25 09:45:00 | 2527.40 | 2025-03-04 14:15:00 | 2555.95 | STOP_HIT | 0.50 | -1.13% |
| SELL | retest2 | 2025-02-27 09:30:00 | 2530.25 | 2025-03-04 14:15:00 | 2555.95 | STOP_HIT | 0.50 | -1.02% |
| BUY | retest2 | 2025-04-02 10:15:00 | 2799.55 | 2025-04-04 12:15:00 | 2667.00 | STOP_HIT | 1.00 | -4.73% |
| BUY | retest2 | 2025-04-02 10:45:00 | 2806.35 | 2025-04-04 12:15:00 | 2667.00 | STOP_HIT | 1.00 | -4.97% |
| BUY | retest2 | 2025-04-02 12:15:00 | 2795.00 | 2025-04-04 12:15:00 | 2667.00 | STOP_HIT | 1.00 | -4.58% |
| BUY | retest2 | 2025-04-02 13:15:00 | 2806.00 | 2025-04-04 12:15:00 | 2667.00 | STOP_HIT | 1.00 | -4.95% |
| BUY | retest2 | 2025-04-16 12:15:00 | 2809.00 | 2025-05-07 12:15:00 | 2680.00 | STOP_HIT | 1.00 | -4.59% |
| BUY | retest2 | 2025-04-16 13:00:00 | 2830.10 | 2025-05-07 12:15:00 | 2680.00 | STOP_HIT | 1.00 | -5.30% |
| BUY | retest2 | 2025-04-29 15:15:00 | 2806.50 | 2025-05-07 12:15:00 | 2680.00 | STOP_HIT | 1.00 | -4.51% |
| BUY | retest2 | 2025-05-06 09:30:00 | 2812.60 | 2025-05-07 12:15:00 | 2680.00 | STOP_HIT | 1.00 | -4.71% |
| SELL | retest2 | 2025-08-18 11:45:00 | 2488.40 | 2025-09-02 09:15:00 | 2486.10 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2025-08-18 15:00:00 | 2485.00 | 2025-09-02 09:15:00 | 2486.10 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-08-22 09:15:00 | 2477.20 | 2025-09-02 09:15:00 | 2486.10 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-08-25 12:00:00 | 2495.60 | 2025-09-02 09:15:00 | 2486.10 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2025-08-26 11:30:00 | 2455.00 | 2025-09-02 09:15:00 | 2486.10 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-08-26 14:15:00 | 2459.70 | 2025-09-02 09:15:00 | 2486.10 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-08-26 15:00:00 | 2458.70 | 2025-09-02 09:15:00 | 2486.10 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-08-28 09:15:00 | 2435.70 | 2025-09-02 09:15:00 | 2486.10 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-08-29 09:15:00 | 2436.10 | 2025-09-03 09:15:00 | 2503.40 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2025-09-01 10:00:00 | 2436.20 | 2025-09-03 10:15:00 | 2522.60 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2025-09-01 11:00:00 | 2440.60 | 2025-09-03 10:15:00 | 2522.60 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2025-09-01 14:30:00 | 2441.50 | 2025-09-03 10:15:00 | 2522.60 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2025-09-02 13:45:00 | 2475.30 | 2025-09-03 10:15:00 | 2522.60 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2026-02-13 09:15:00 | 2376.70 | 2026-03-02 09:15:00 | 2257.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-13 09:15:00 | 2376.70 | 2026-03-05 11:15:00 | 2139.03 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-13 09:15:00 | 2378.70 | 2026-04-28 09:15:00 | 2276.49 | PARTIAL | 0.50 | 4.30% |
| SELL | retest2 | 2026-04-15 11:00:00 | 2396.30 | 2026-04-28 09:15:00 | 2282.85 | PARTIAL | 0.50 | 4.73% |
| SELL | retest2 | 2026-04-13 09:15:00 | 2378.70 | 2026-05-04 10:15:00 | 2344.90 | STOP_HIT | 0.50 | 1.42% |
| SELL | retest2 | 2026-04-15 11:00:00 | 2396.30 | 2026-05-04 10:15:00 | 2344.90 | STOP_HIT | 0.50 | 2.14% |
| SELL | retest2 | 2026-04-17 10:15:00 | 2403.00 | 2026-05-06 09:15:00 | 2350.20 | STOP_HIT | 1.00 | 2.20% |
| SELL | retest2 | 2026-05-05 09:15:00 | 2313.00 | 2026-05-06 09:15:00 | 2350.20 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-05-05 11:00:00 | 2332.50 | 2026-05-08 10:15:00 | 2353.70 | STOP_HIT | 1.00 | -0.91% |
