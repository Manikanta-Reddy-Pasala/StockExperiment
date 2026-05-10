# Balkrishna Industries Ltd. (BALKRISIND)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 2265.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 68 |
| ALERT1 | 44 |
| ALERT2 | 44 |
| ALERT2_SKIP | 20 |
| ALERT3 | 108 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 56 |
| PARTIAL | 8 |
| TARGET_HIT | 3 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 66 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 27 / 39
- **Target hits / Stop hits / Partials:** 3 / 55 / 8
- **Avg / median % per leg:** 0.79% / -0.58%
- **Sum % (uncompounded):** 52.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 8 | 42.1% | 1 | 18 | 0 | 0.40% | 7.5% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.03% | -2.0% |
| BUY @ 3rd Alert (retest2) | 18 | 8 | 44.4% | 1 | 17 | 0 | 0.53% | 9.6% |
| SELL (all) | 47 | 19 | 40.4% | 2 | 37 | 8 | 0.95% | 44.6% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.21% | -1.2% |
| SELL @ 3rd Alert (retest2) | 46 | 19 | 41.3% | 2 | 36 | 8 | 1.00% | 45.8% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.62% | -3.2% |
| retest2 (combined) | 64 | 27 | 42.2% | 3 | 53 | 8 | 0.87% | 55.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 10:15:00 | 2726.00 | 2745.30 | 2746.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 11:15:00 | 2722.70 | 2740.78 | 2744.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-14 09:15:00 | 2729.40 | 2720.08 | 2730.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 2729.40 | 2720.08 | 2730.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 2729.40 | 2720.08 | 2730.53 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 09:15:00 | 2749.70 | 2736.66 | 2735.14 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 13:15:00 | 2708.60 | 2730.67 | 2732.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-15 15:15:00 | 2690.00 | 2717.90 | 2726.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 14:15:00 | 2712.80 | 2707.27 | 2715.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 14:15:00 | 2712.80 | 2707.27 | 2715.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 2712.80 | 2707.27 | 2715.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 15:00:00 | 2712.80 | 2707.27 | 2715.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 2749.40 | 2716.93 | 2718.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:00:00 | 2749.40 | 2716.93 | 2718.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 10:15:00 | 2735.80 | 2720.71 | 2720.45 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 13:15:00 | 2694.20 | 2716.32 | 2718.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 14:15:00 | 2690.30 | 2711.11 | 2716.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 2679.50 | 2664.57 | 2682.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 10:00:00 | 2679.50 | 2664.57 | 2682.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 2660.30 | 2663.72 | 2680.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:30:00 | 2682.00 | 2663.72 | 2680.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 2653.00 | 2659.68 | 2675.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:30:00 | 2674.30 | 2659.68 | 2675.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 15:15:00 | 2655.90 | 2659.28 | 2671.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:15:00 | 2668.40 | 2659.28 | 2671.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 2665.90 | 2660.61 | 2670.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 12:15:00 | 2634.00 | 2657.84 | 2667.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 13:30:00 | 2632.00 | 2648.75 | 2661.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 11:15:00 | 2641.10 | 2655.05 | 2660.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 12:00:00 | 2641.70 | 2652.38 | 2658.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 2653.80 | 2649.72 | 2656.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 13:45:00 | 2653.90 | 2649.72 | 2656.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 2656.70 | 2651.12 | 2656.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 14:30:00 | 2664.00 | 2651.12 | 2656.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 2671.00 | 2655.09 | 2657.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 09:15:00 | 2397.00 | 2655.09 | 2657.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-26 09:15:00 | 2502.30 | 2610.56 | 2637.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-26 09:15:00 | 2500.40 | 2610.56 | 2637.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-26 09:15:00 | 2509.04 | 2610.56 | 2637.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-26 09:15:00 | 2509.61 | 2610.56 | 2637.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-28 13:15:00 | 2476.30 | 2476.16 | 2507.67 | SL hit (close>ema200) qty=0.50 sl=2476.16 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-28 13:15:00 | 2476.30 | 2476.16 | 2507.67 | SL hit (close>ema200) qty=0.50 sl=2476.16 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-28 13:15:00 | 2476.30 | 2476.16 | 2507.67 | SL hit (close>ema200) qty=0.50 sl=2476.16 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-28 13:15:00 | 2476.30 | 2476.16 | 2507.67 | SL hit (close>ema200) qty=0.50 sl=2476.16 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-05 09:15:00 | 2471.70 | 2460.27 | 2459.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 2471.70 | 2460.27 | 2459.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 09:15:00 | 2489.70 | 2468.77 | 2464.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 11:15:00 | 2511.40 | 2515.95 | 2505.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 12:00:00 | 2511.40 | 2515.95 | 2505.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 2500.00 | 2512.76 | 2505.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:00:00 | 2500.00 | 2512.76 | 2505.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 2494.90 | 2509.19 | 2504.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 2497.00 | 2509.19 | 2504.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 2497.50 | 2504.95 | 2502.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 2494.90 | 2504.95 | 2502.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 2477.50 | 2497.56 | 2499.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 14:15:00 | 2469.50 | 2482.56 | 2491.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 2465.80 | 2464.18 | 2474.65 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:15:00 | 2440.00 | 2464.18 | 2474.65 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 2462.80 | 2455.32 | 2464.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 2462.80 | 2455.32 | 2464.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 2460.90 | 2457.14 | 2463.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-17 10:15:00 | 2469.60 | 2459.63 | 2464.05 | SL hit (close>ema400) qty=1.00 sl=2464.05 alert=retest1 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 09:30:00 | 2442.80 | 2452.77 | 2458.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:15:00 | 2440.30 | 2452.77 | 2458.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:15:00 | 2442.00 | 2449.79 | 2455.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 12:45:00 | 2437.60 | 2447.06 | 2450.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 2417.90 | 2420.07 | 2429.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:45:00 | 2433.00 | 2420.07 | 2429.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 2412.50 | 2405.33 | 2413.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 13:15:00 | 2400.00 | 2408.85 | 2413.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 09:30:00 | 2400.50 | 2401.55 | 2408.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 10:45:00 | 2402.90 | 2402.14 | 2407.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 12:15:00 | 2432.00 | 2408.49 | 2406.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-26 12:15:00 | 2432.00 | 2408.49 | 2406.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-26 12:15:00 | 2432.00 | 2408.49 | 2406.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-26 12:15:00 | 2432.00 | 2408.49 | 2406.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-26 12:15:00 | 2432.00 | 2408.49 | 2406.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-26 12:15:00 | 2432.00 | 2408.49 | 2406.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-26 12:15:00 | 2432.00 | 2408.49 | 2406.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 12:15:00 | 2432.00 | 2408.49 | 2406.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 2434.90 | 2423.33 | 2415.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 13:15:00 | 2460.00 | 2468.73 | 2451.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 13:15:00 | 2460.00 | 2468.73 | 2451.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 2460.00 | 2468.73 | 2451.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:45:00 | 2456.00 | 2468.73 | 2451.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 2449.60 | 2464.90 | 2450.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 15:00:00 | 2449.60 | 2464.90 | 2450.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 2440.00 | 2459.92 | 2449.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 10:00:00 | 2458.00 | 2459.54 | 2450.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-17 09:15:00 | 2703.80 | 2667.59 | 2659.49 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 12:15:00 | 2724.50 | 2742.45 | 2742.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 2688.80 | 2725.90 | 2733.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 2705.00 | 2695.72 | 2710.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 10:00:00 | 2705.00 | 2695.72 | 2710.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 2701.20 | 2696.82 | 2710.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:30:00 | 2736.40 | 2696.82 | 2710.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 2707.10 | 2698.87 | 2709.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:45:00 | 2722.20 | 2698.87 | 2709.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 12:15:00 | 2700.10 | 2699.12 | 2708.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 12:30:00 | 2700.30 | 2699.12 | 2708.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 2726.40 | 2702.44 | 2707.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:45:00 | 2719.20 | 2702.44 | 2707.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 2724.70 | 2706.89 | 2708.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:30:00 | 2732.00 | 2706.89 | 2708.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2025-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 11:15:00 | 2728.30 | 2711.17 | 2710.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 12:15:00 | 2741.10 | 2717.16 | 2713.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 2704.40 | 2738.94 | 2732.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 2704.40 | 2738.94 | 2732.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 2704.40 | 2738.94 | 2732.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:15:00 | 2704.70 | 2738.94 | 2732.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 11:15:00 | 2696.60 | 2724.47 | 2726.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 12:15:00 | 2688.60 | 2717.29 | 2722.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 15:15:00 | 2585.00 | 2583.97 | 2615.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-05 09:15:00 | 2575.20 | 2583.97 | 2615.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 2412.70 | 2424.66 | 2440.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 10:30:00 | 2396.20 | 2419.99 | 2436.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 11:15:00 | 2400.10 | 2419.99 | 2436.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 12:00:00 | 2400.10 | 2416.01 | 2433.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 2390.40 | 2413.69 | 2426.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 2402.20 | 2411.39 | 2424.19 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-18 10:15:00 | 2440.00 | 2412.69 | 2411.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-18 10:15:00 | 2440.00 | 2412.69 | 2411.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-18 10:15:00 | 2440.00 | 2412.69 | 2411.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-18 10:15:00 | 2440.00 | 2412.69 | 2411.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 2440.00 | 2412.69 | 2411.79 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 10:15:00 | 2413.60 | 2420.34 | 2420.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 11:15:00 | 2402.50 | 2416.77 | 2419.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 13:15:00 | 2420.40 | 2415.60 | 2418.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 13:15:00 | 2420.40 | 2415.60 | 2418.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 2420.40 | 2415.60 | 2418.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:00:00 | 2420.40 | 2415.60 | 2418.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 2405.20 | 2413.52 | 2417.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 09:45:00 | 2401.10 | 2412.87 | 2416.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 2281.04 | 2315.75 | 2337.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 11:15:00 | 2291.90 | 2287.46 | 2305.94 | SL hit (close>ema200) qty=0.50 sl=2287.46 alert=retest2 |

### Cycle 14 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 2351.80 | 2314.54 | 2313.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 12:15:00 | 2374.90 | 2335.27 | 2323.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 2350.40 | 2352.15 | 2337.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 11:00:00 | 2350.40 | 2352.15 | 2337.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 2366.90 | 2354.43 | 2341.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 12:30:00 | 2350.00 | 2354.43 | 2341.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 2349.40 | 2361.50 | 2352.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:00:00 | 2349.40 | 2361.50 | 2352.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 2345.00 | 2358.20 | 2351.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:00:00 | 2345.00 | 2358.20 | 2351.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 2338.40 | 2354.24 | 2350.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 2338.40 | 2354.24 | 2350.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — SELL (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 09:15:00 | 2334.90 | 2347.76 | 2348.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 2310.00 | 2337.45 | 2343.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 2350.90 | 2328.23 | 2335.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 2350.90 | 2328.23 | 2335.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 2350.90 | 2328.23 | 2335.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 2350.90 | 2328.23 | 2335.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 2328.40 | 2328.27 | 2334.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 11:15:00 | 2320.40 | 2328.27 | 2334.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 12:15:00 | 2323.40 | 2328.19 | 2333.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 13:15:00 | 2320.30 | 2327.65 | 2333.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 14:30:00 | 2322.70 | 2325.88 | 2331.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 2311.60 | 2322.66 | 2328.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 10:30:00 | 2306.80 | 2319.83 | 2327.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:00:00 | 2308.50 | 2319.83 | 2327.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 12:30:00 | 2307.60 | 2315.29 | 2323.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 2349.50 | 2323.90 | 2323.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 2349.50 | 2323.90 | 2323.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 2349.50 | 2323.90 | 2323.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 2349.50 | 2323.90 | 2323.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 2349.50 | 2323.90 | 2323.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 2349.50 | 2323.90 | 2323.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 2349.50 | 2323.90 | 2323.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 11:15:00 | 2349.50 | 2323.90 | 2323.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 13:15:00 | 2361.80 | 2336.01 | 2329.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 13:15:00 | 2394.90 | 2409.89 | 2386.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 14:00:00 | 2394.90 | 2409.89 | 2386.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 2388.50 | 2405.61 | 2387.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 15:15:00 | 2387.00 | 2405.61 | 2387.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 2387.00 | 2401.89 | 2387.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 09:15:00 | 2406.00 | 2401.89 | 2387.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 15:15:00 | 2489.00 | 2503.97 | 2504.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2025-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 15:15:00 | 2489.00 | 2503.97 | 2504.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 2455.40 | 2494.26 | 2500.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 11:15:00 | 2440.50 | 2433.22 | 2455.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 12:00:00 | 2440.50 | 2433.22 | 2455.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 2307.30 | 2300.30 | 2312.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 11:15:00 | 2292.10 | 2301.29 | 2306.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 12:30:00 | 2289.40 | 2298.10 | 2304.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 11:30:00 | 2289.50 | 2295.94 | 2300.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 11:15:00 | 2320.10 | 2296.36 | 2296.79 | SL hit (close>static) qty=1.00 sl=2317.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-08 11:15:00 | 2320.10 | 2296.36 | 2296.79 | SL hit (close>static) qty=1.00 sl=2317.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-08 11:15:00 | 2320.10 | 2296.36 | 2296.79 | SL hit (close>static) qty=1.00 sl=2317.00 alert=retest2 |

### Cycle 18 — BUY (started 2025-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 12:15:00 | 2302.00 | 2297.49 | 2297.26 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2025-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 13:15:00 | 2295.00 | 2296.99 | 2297.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 2281.80 | 2293.95 | 2295.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 2297.90 | 2293.25 | 2294.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 2297.90 | 2293.25 | 2294.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 2297.90 | 2293.25 | 2294.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:00:00 | 2297.90 | 2293.25 | 2294.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 2270.30 | 2288.66 | 2292.71 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 2295.80 | 2290.23 | 2290.18 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 14:15:00 | 2287.10 | 2289.71 | 2289.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 2247.30 | 2280.83 | 2285.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 10:15:00 | 2233.00 | 2232.41 | 2252.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-14 11:00:00 | 2233.00 | 2232.41 | 2252.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 2220.00 | 2228.22 | 2244.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:45:00 | 2242.00 | 2228.22 | 2244.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 2235.30 | 2222.71 | 2231.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:45:00 | 2238.10 | 2222.71 | 2231.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 2233.40 | 2224.85 | 2231.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:45:00 | 2231.00 | 2224.85 | 2231.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 2238.00 | 2227.48 | 2231.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:30:00 | 2230.10 | 2227.48 | 2231.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 2243.70 | 2230.72 | 2232.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:30:00 | 2245.80 | 2230.72 | 2232.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 2263.00 | 2237.18 | 2235.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 14:15:00 | 2268.30 | 2243.40 | 2238.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 15:15:00 | 2265.00 | 2265.85 | 2256.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 09:15:00 | 2272.30 | 2265.85 | 2256.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 2278.00 | 2286.97 | 2275.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 2301.50 | 2289.77 | 2277.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 12:15:00 | 2322.10 | 2329.00 | 2329.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2025-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 12:15:00 | 2322.10 | 2329.00 | 2329.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 13:15:00 | 2310.20 | 2325.24 | 2327.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 15:15:00 | 2330.00 | 2326.04 | 2327.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 15:15:00 | 2330.00 | 2326.04 | 2327.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 2330.00 | 2326.04 | 2327.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:15:00 | 2330.00 | 2326.04 | 2327.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 2335.00 | 2327.83 | 2328.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 2335.00 | 2327.83 | 2328.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 2327.10 | 2327.69 | 2328.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:45:00 | 2330.90 | 2327.69 | 2328.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 2328.70 | 2327.89 | 2328.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:00:00 | 2328.70 | 2327.89 | 2328.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 2303.30 | 2322.97 | 2326.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 14:15:00 | 2290.10 | 2318.40 | 2323.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:15:00 | 2175.59 | 2293.18 | 2309.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-03 14:15:00 | 2290.40 | 2283.07 | 2297.14 | SL hit (close>ema200) qty=0.50 sl=2283.07 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 15:00:00 | 2290.40 | 2283.07 | 2297.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 2347.00 | 2298.20 | 2301.72 | SL hit (close>static) qty=1.00 sl=2330.80 alert=retest2 |

### Cycle 24 — BUY (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 10:15:00 | 2347.90 | 2308.14 | 2305.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 11:15:00 | 2355.10 | 2317.53 | 2310.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 2299.90 | 2334.45 | 2323.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 2299.90 | 2334.45 | 2323.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 2299.90 | 2334.45 | 2323.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 2299.90 | 2334.45 | 2323.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 2330.50 | 2333.66 | 2324.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 11:15:00 | 2337.20 | 2333.66 | 2324.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 11:15:00 | 2315.60 | 2325.13 | 2325.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2025-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 11:15:00 | 2315.60 | 2325.13 | 2325.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 15:15:00 | 2301.00 | 2315.11 | 2318.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 11:15:00 | 2315.80 | 2311.65 | 2315.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-11 12:00:00 | 2315.80 | 2311.65 | 2315.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 2331.90 | 2315.70 | 2317.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 12:45:00 | 2329.90 | 2315.70 | 2317.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2025-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 13:15:00 | 2342.90 | 2321.14 | 2319.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 14:15:00 | 2347.10 | 2326.33 | 2322.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 12:15:00 | 2363.50 | 2368.71 | 2355.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 12:30:00 | 2360.30 | 2368.71 | 2355.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 2331.80 | 2360.28 | 2354.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 2331.80 | 2360.28 | 2354.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 2340.30 | 2356.29 | 2352.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 2332.90 | 2356.29 | 2352.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 2323.20 | 2349.67 | 2350.28 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 11:15:00 | 2361.70 | 2347.22 | 2346.19 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 2316.30 | 2347.67 | 2347.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 10:15:00 | 2308.10 | 2339.75 | 2344.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 10:15:00 | 2317.90 | 2316.91 | 2327.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 10:15:00 | 2317.90 | 2316.91 | 2327.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 2317.90 | 2316.91 | 2327.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:45:00 | 2319.20 | 2316.91 | 2327.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 2322.20 | 2317.16 | 2324.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 15:00:00 | 2322.20 | 2317.16 | 2324.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 2321.00 | 2317.92 | 2324.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 2317.80 | 2317.92 | 2324.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 2287.10 | 2311.76 | 2320.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 10:30:00 | 2269.90 | 2303.37 | 2316.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 13:00:00 | 2269.10 | 2291.14 | 2307.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 2339.80 | 2302.13 | 2307.73 | SL hit (close>static) qty=1.00 sl=2336.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 2339.80 | 2302.13 | 2307.73 | SL hit (close>static) qty=1.00 sl=2336.00 alert=retest2 |

### Cycle 30 — BUY (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 11:15:00 | 2341.00 | 2313.82 | 2312.31 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 12:15:00 | 2281.60 | 2311.23 | 2313.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 14:15:00 | 2263.10 | 2297.34 | 2306.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 13:15:00 | 2287.30 | 2278.81 | 2291.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 14:00:00 | 2287.30 | 2278.81 | 2291.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 2309.40 | 2284.93 | 2292.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 2309.40 | 2284.93 | 2292.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 2303.00 | 2288.54 | 2293.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 2274.90 | 2288.54 | 2293.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 2312.40 | 2297.12 | 2296.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 12:15:00 | 2324.20 | 2305.25 | 2300.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 2299.30 | 2307.31 | 2303.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 10:15:00 | 2299.30 | 2307.31 | 2303.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 2299.30 | 2307.31 | 2303.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:00:00 | 2299.30 | 2307.31 | 2303.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 2295.20 | 2304.89 | 2303.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:45:00 | 2296.10 | 2304.89 | 2303.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 13:15:00 | 2298.20 | 2301.97 | 2302.01 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2025-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 14:15:00 | 2303.80 | 2302.34 | 2302.17 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2025-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 15:15:00 | 2300.00 | 2301.87 | 2301.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 2268.00 | 2295.09 | 2298.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 12:15:00 | 2294.70 | 2288.81 | 2294.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 12:15:00 | 2294.70 | 2288.81 | 2294.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 2294.70 | 2288.81 | 2294.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 12:30:00 | 2298.10 | 2288.81 | 2294.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 2296.70 | 2290.39 | 2294.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 13:30:00 | 2296.40 | 2290.39 | 2294.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 2315.10 | 2295.33 | 2296.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 15:00:00 | 2315.10 | 2295.33 | 2296.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2025-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 15:15:00 | 2310.00 | 2298.27 | 2297.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 09:15:00 | 2368.20 | 2314.76 | 2306.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 14:15:00 | 2409.90 | 2411.98 | 2383.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-03 15:00:00 | 2409.90 | 2411.98 | 2383.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 2395.00 | 2405.39 | 2385.44 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 14:15:00 | 2379.80 | 2388.31 | 2389.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 2356.00 | 2382.76 | 2386.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 15:15:00 | 2324.90 | 2323.69 | 2336.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 09:15:00 | 2313.50 | 2323.69 | 2336.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 2315.80 | 2313.10 | 2324.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:00:00 | 2315.80 | 2313.10 | 2324.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 2317.60 | 2313.60 | 2322.85 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2025-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 14:15:00 | 2350.10 | 2329.30 | 2327.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 11:15:00 | 2357.00 | 2339.88 | 2333.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 12:15:00 | 2369.30 | 2370.36 | 2357.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 12:45:00 | 2370.00 | 2370.36 | 2357.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 2350.30 | 2385.66 | 2378.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:30:00 | 2348.00 | 2385.66 | 2378.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 2350.50 | 2378.63 | 2375.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:00:00 | 2350.50 | 2378.63 | 2375.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2025-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 11:15:00 | 2350.40 | 2372.98 | 2373.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-22 09:15:00 | 2346.00 | 2357.97 | 2363.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 09:15:00 | 2303.00 | 2299.16 | 2306.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 2303.00 | 2299.16 | 2306.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 2303.00 | 2299.16 | 2306.24 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 2309.50 | 2305.22 | 2305.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 14:15:00 | 2322.70 | 2308.72 | 2306.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 15:15:00 | 2306.60 | 2308.29 | 2306.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 15:15:00 | 2306.60 | 2308.29 | 2306.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 2306.60 | 2308.29 | 2306.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 14:15:00 | 2323.90 | 2315.79 | 2311.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 14:45:00 | 2324.90 | 2316.73 | 2312.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 15:15:00 | 2306.00 | 2314.58 | 2311.62 | SL hit (close<static) qty=1.00 sl=2306.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-01 15:15:00 | 2306.00 | 2314.58 | 2311.62 | SL hit (close<static) qty=1.00 sl=2306.60 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 2327.00 | 2314.58 | 2311.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 09:15:00 | 2371.70 | 2380.91 | 2381.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 2371.70 | 2380.91 | 2381.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 12:15:00 | 2349.20 | 2371.92 | 2376.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 15:15:00 | 2368.00 | 2365.56 | 2372.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 15:15:00 | 2368.00 | 2365.56 | 2372.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 15:15:00 | 2368.00 | 2365.56 | 2372.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:15:00 | 2359.40 | 2365.56 | 2372.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 2383.70 | 2369.19 | 2373.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:30:00 | 2385.80 | 2369.19 | 2373.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 2386.70 | 2372.69 | 2374.49 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2026-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 11:15:00 | 2394.40 | 2377.03 | 2376.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 12:15:00 | 2400.00 | 2381.62 | 2378.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 09:15:00 | 2395.80 | 2396.92 | 2387.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:30:00 | 2390.30 | 2396.92 | 2387.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 2399.60 | 2400.70 | 2393.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:30:00 | 2388.00 | 2400.70 | 2393.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 2411.00 | 2402.00 | 2395.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 11:30:00 | 2421.00 | 2405.19 | 2397.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 12:30:00 | 2421.20 | 2404.15 | 2398.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 14:45:00 | 2432.90 | 2408.79 | 2401.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 10:30:00 | 2420.70 | 2413.79 | 2405.70 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 2439.60 | 2432.10 | 2422.84 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-20 10:15:00 | 2407.10 | 2418.01 | 2418.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-20 10:15:00 | 2407.10 | 2418.01 | 2418.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-20 10:15:00 | 2407.10 | 2418.01 | 2418.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-20 10:15:00 | 2407.10 | 2418.01 | 2418.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2026-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 10:15:00 | 2407.10 | 2418.01 | 2418.65 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 09:15:00 | 2425.00 | 2418.21 | 2417.31 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 2412.00 | 2417.78 | 2418.00 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 09:15:00 | 2450.90 | 2420.27 | 2418.34 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2026-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 14:15:00 | 2412.30 | 2418.29 | 2418.56 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 2433.00 | 2421.46 | 2419.97 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 14:15:00 | 2411.80 | 2419.62 | 2419.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 09:15:00 | 2385.00 | 2410.68 | 2415.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 2309.40 | 2305.41 | 2326.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 2309.40 | 2305.41 | 2326.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 2309.40 | 2305.41 | 2326.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:00:00 | 2309.40 | 2305.41 | 2326.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 2294.90 | 2278.63 | 2302.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 2294.90 | 2278.63 | 2302.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 2304.00 | 2283.70 | 2302.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 2474.20 | 2283.70 | 2302.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2424.90 | 2311.94 | 2313.37 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 2497.50 | 2349.05 | 2330.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 2529.00 | 2385.04 | 2348.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 10:15:00 | 2600.10 | 2601.02 | 2530.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:45:00 | 2602.50 | 2601.02 | 2530.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 2605.00 | 2642.05 | 2594.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 2592.60 | 2642.05 | 2594.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 2681.20 | 2706.40 | 2687.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 2681.20 | 2706.40 | 2687.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 2684.40 | 2702.00 | 2687.43 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 2661.80 | 2678.25 | 2680.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 10:15:00 | 2637.50 | 2670.10 | 2676.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 2579.10 | 2577.70 | 2601.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 2579.10 | 2577.70 | 2601.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 2591.40 | 2549.80 | 2557.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 2591.40 | 2549.80 | 2557.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 2595.10 | 2558.86 | 2560.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 2595.10 | 2558.86 | 2560.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 11:15:00 | 2589.50 | 2564.99 | 2563.25 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 2536.00 | 2561.56 | 2562.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 09:15:00 | 2514.60 | 2547.12 | 2555.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 13:15:00 | 2507.90 | 2507.58 | 2522.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-23 14:00:00 | 2507.90 | 2507.58 | 2522.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 2531.20 | 2502.10 | 2510.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 15:00:00 | 2531.20 | 2502.10 | 2510.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 2508.00 | 2503.28 | 2510.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 14:00:00 | 2489.10 | 2503.21 | 2507.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 15:15:00 | 2489.00 | 2506.01 | 2508.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2364.64 | 2391.52 | 2430.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2364.55 | 2391.52 | 2430.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-05 10:15:00 | 2240.19 | 2261.26 | 2304.65 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-05 10:15:00 | 2240.10 | 2261.26 | 2304.65 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 54 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 2253.50 | 2246.80 | 2246.78 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 09:15:00 | 2233.50 | 2244.14 | 2245.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 10:15:00 | 2225.00 | 2240.31 | 2243.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 12:15:00 | 2225.00 | 2203.70 | 2216.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 12:15:00 | 2225.00 | 2203.70 | 2216.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 2225.00 | 2203.70 | 2216.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:45:00 | 2226.70 | 2203.70 | 2216.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 2241.30 | 2211.22 | 2218.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:45:00 | 2240.00 | 2211.22 | 2218.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 14:15:00 | 2286.20 | 2226.22 | 2225.10 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2026-03-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 09:15:00 | 2132.40 | 2212.57 | 2223.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 2120.20 | 2146.52 | 2157.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 2059.70 | 2050.91 | 2075.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:45:00 | 2057.00 | 2050.91 | 2075.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 2063.00 | 2053.33 | 2074.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 2079.30 | 2053.33 | 2074.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 2059.80 | 2054.62 | 2072.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 15:15:00 | 2050.00 | 2054.62 | 2072.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 2143.50 | 2071.66 | 2077.42 | SL hit (close>static) qty=1.00 sl=2073.90 alert=retest2 |

### Cycle 58 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 2153.90 | 2088.11 | 2084.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 2154.60 | 2101.40 | 2090.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 2146.50 | 2150.50 | 2122.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 10:00:00 | 2146.50 | 2150.50 | 2122.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 2130.40 | 2143.55 | 2124.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 13:15:00 | 2145.90 | 2141.56 | 2124.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 2074.60 | 2133.41 | 2127.14 | SL hit (close<static) qty=1.00 sl=2118.00 alert=retest2 |

### Cycle 59 — SELL (started 2026-03-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 11:15:00 | 2088.40 | 2117.14 | 2120.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 12:15:00 | 2078.40 | 2109.39 | 2116.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 2108.80 | 2097.29 | 2107.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 2108.80 | 2097.29 | 2107.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 2108.80 | 2097.29 | 2107.26 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 15:15:00 | 2130.00 | 2112.87 | 2111.17 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 2027.00 | 2095.70 | 2103.52 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 2118.10 | 2086.19 | 2085.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 2221.40 | 2138.37 | 2115.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 2194.30 | 2198.54 | 2164.14 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 2274.40 | 2217.40 | 2190.49 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2228.30 | 2251.95 | 2228.60 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 2228.30 | 2251.95 | 2228.60 | SL hit (close<ema400) qty=1.00 sl=2228.60 alert=retest1 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:00:00 | 2258.40 | 2253.24 | 2231.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:00:00 | 2249.50 | 2252.49 | 2232.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 14:15:00 | 2249.90 | 2249.84 | 2235.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 2257.30 | 2241.87 | 2233.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 2304.50 | 2323.59 | 2308.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 10:00:00 | 2304.50 | 2323.59 | 2308.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 2311.30 | 2321.14 | 2308.38 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-20 15:15:00 | 2279.80 | 2303.61 | 2304.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-20 15:15:00 | 2279.80 | 2303.61 | 2304.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-20 15:15:00 | 2279.80 | 2303.61 | 2304.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-20 15:15:00 | 2279.80 | 2303.61 | 2304.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 2279.80 | 2303.61 | 2304.10 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 10:15:00 | 2304.70 | 2300.45 | 2300.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 12:15:00 | 2311.30 | 2304.10 | 2301.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 09:15:00 | 2283.00 | 2302.79 | 2302.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 09:15:00 | 2283.00 | 2302.79 | 2302.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 2283.00 | 2302.79 | 2302.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:45:00 | 2288.00 | 2302.79 | 2302.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 2282.00 | 2298.63 | 2300.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 11:15:00 | 2270.10 | 2292.93 | 2297.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 14:15:00 | 2236.20 | 2231.55 | 2256.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 15:00:00 | 2236.20 | 2231.55 | 2256.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 2244.00 | 2235.43 | 2253.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 2246.90 | 2235.43 | 2253.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 2199.70 | 2222.61 | 2237.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 12:00:00 | 2195.00 | 2213.47 | 2230.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 2175.90 | 2212.49 | 2218.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 11:30:00 | 2195.20 | 2180.85 | 2189.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:30:00 | 2187.80 | 2183.20 | 2189.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 13:15:00 | 2195.20 | 2185.60 | 2190.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 13:45:00 | 2196.00 | 2185.60 | 2190.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 2204.80 | 2195.01 | 2193.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 2204.80 | 2195.01 | 2193.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 2204.80 | 2195.01 | 2193.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 2204.80 | 2195.01 | 2193.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — BUY (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 09:15:00 | 2204.80 | 2195.01 | 2193.69 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2026-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 14:15:00 | 2177.90 | 2191.35 | 2192.91 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 2203.80 | 2193.74 | 2193.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 2218.00 | 2204.10 | 2198.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 14:15:00 | 2256.80 | 2272.50 | 2252.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 14:15:00 | 2256.80 | 2272.50 | 2252.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 2256.80 | 2272.50 | 2252.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:00:00 | 2256.80 | 2272.50 | 2252.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 2265.10 | 2271.02 | 2253.77 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 09:15:00 | 2772.10 | 2025-05-13 10:15:00 | 2726.00 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-05-12 15:15:00 | 2755.00 | 2025-05-13 10:15:00 | 2726.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-05-22 12:15:00 | 2634.00 | 2025-05-26 09:15:00 | 2502.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-22 13:30:00 | 2632.00 | 2025-05-26 09:15:00 | 2500.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-23 11:15:00 | 2641.10 | 2025-05-26 09:15:00 | 2509.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-23 12:00:00 | 2641.70 | 2025-05-26 09:15:00 | 2509.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-22 12:15:00 | 2634.00 | 2025-05-28 13:15:00 | 2476.30 | STOP_HIT | 0.50 | 5.99% |
| SELL | retest2 | 2025-05-22 13:30:00 | 2632.00 | 2025-05-28 13:15:00 | 2476.30 | STOP_HIT | 0.50 | 5.92% |
| SELL | retest2 | 2025-05-23 11:15:00 | 2641.10 | 2025-05-28 13:15:00 | 2476.30 | STOP_HIT | 0.50 | 6.24% |
| SELL | retest2 | 2025-05-23 12:00:00 | 2641.70 | 2025-05-28 13:15:00 | 2476.30 | STOP_HIT | 0.50 | 6.26% |
| SELL | retest2 | 2025-05-26 09:15:00 | 2397.00 | 2025-06-05 09:15:00 | 2471.70 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest1 | 2025-06-16 09:15:00 | 2440.00 | 2025-06-17 10:15:00 | 2469.60 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-06-18 09:30:00 | 2442.80 | 2025-06-26 12:15:00 | 2432.00 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2025-06-18 10:15:00 | 2440.30 | 2025-06-26 12:15:00 | 2432.00 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2025-06-18 12:15:00 | 2442.00 | 2025-06-26 12:15:00 | 2432.00 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2025-06-19 12:45:00 | 2437.60 | 2025-06-26 12:15:00 | 2432.00 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2025-06-24 13:15:00 | 2400.00 | 2025-06-26 12:15:00 | 2432.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-06-25 09:30:00 | 2400.50 | 2025-06-26 12:15:00 | 2432.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-06-25 10:45:00 | 2402.90 | 2025-06-26 12:15:00 | 2432.00 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-07-01 10:00:00 | 2458.00 | 2025-07-17 09:15:00 | 2703.80 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-11 10:30:00 | 2396.20 | 2025-08-18 10:15:00 | 2440.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-08-11 11:15:00 | 2400.10 | 2025-08-18 10:15:00 | 2440.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-08-11 12:00:00 | 2400.10 | 2025-08-18 10:15:00 | 2440.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-08-12 09:15:00 | 2390.40 | 2025-08-18 10:15:00 | 2440.00 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-08-22 09:45:00 | 2401.10 | 2025-08-29 09:15:00 | 2281.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-22 09:45:00 | 2401.10 | 2025-09-01 11:15:00 | 2291.90 | STOP_HIT | 0.50 | 4.55% |
| SELL | retest2 | 2025-09-08 11:15:00 | 2320.40 | 2025-09-10 11:15:00 | 2349.50 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-09-08 12:15:00 | 2323.40 | 2025-09-10 11:15:00 | 2349.50 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-09-08 13:15:00 | 2320.30 | 2025-09-10 11:15:00 | 2349.50 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-09-08 14:30:00 | 2322.70 | 2025-09-10 11:15:00 | 2349.50 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-09-09 10:30:00 | 2306.80 | 2025-09-10 11:15:00 | 2349.50 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-09-09 11:00:00 | 2308.50 | 2025-09-10 11:15:00 | 2349.50 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-09-09 12:30:00 | 2307.60 | 2025-09-10 11:15:00 | 2349.50 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-09-15 09:15:00 | 2406.00 | 2025-09-23 15:15:00 | 2489.00 | STOP_HIT | 1.00 | 3.45% |
| SELL | retest2 | 2025-10-06 11:15:00 | 2292.10 | 2025-10-08 11:15:00 | 2320.10 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-10-06 12:30:00 | 2289.40 | 2025-10-08 11:15:00 | 2320.10 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-10-07 11:30:00 | 2289.50 | 2025-10-08 11:15:00 | 2320.10 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-10-21 13:45:00 | 2301.50 | 2025-10-30 12:15:00 | 2322.10 | STOP_HIT | 1.00 | 0.90% |
| SELL | retest2 | 2025-10-31 14:15:00 | 2290.10 | 2025-11-03 09:15:00 | 2175.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 14:15:00 | 2290.10 | 2025-11-03 14:15:00 | 2290.40 | STOP_HIT | 0.50 | -0.01% |
| SELL | retest2 | 2025-11-03 15:00:00 | 2290.40 | 2025-11-04 09:15:00 | 2347.00 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-11-06 11:15:00 | 2337.20 | 2025-11-07 11:15:00 | 2315.60 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-11-20 10:30:00 | 2269.90 | 2025-11-21 09:15:00 | 2339.80 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-11-20 13:00:00 | 2269.10 | 2025-11-21 09:15:00 | 2339.80 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2026-01-01 14:15:00 | 2323.90 | 2026-01-01 15:15:00 | 2306.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2026-01-01 14:45:00 | 2324.90 | 2026-01-01 15:15:00 | 2306.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2026-01-02 09:15:00 | 2327.00 | 2026-01-09 09:15:00 | 2371.70 | STOP_HIT | 1.00 | 1.92% |
| BUY | retest2 | 2026-01-14 11:30:00 | 2421.00 | 2026-01-20 10:15:00 | 2407.10 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2026-01-14 12:30:00 | 2421.20 | 2026-01-20 10:15:00 | 2407.10 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2026-01-14 14:45:00 | 2432.90 | 2026-01-20 10:15:00 | 2407.10 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2026-01-16 10:30:00 | 2420.70 | 2026-01-20 10:15:00 | 2407.10 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2026-02-25 14:00:00 | 2489.10 | 2026-03-02 09:15:00 | 2364.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 15:15:00 | 2489.00 | 2026-03-02 09:15:00 | 2364.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 14:00:00 | 2489.10 | 2026-03-05 10:15:00 | 2240.19 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-25 15:15:00 | 2489.00 | 2026-03-05 10:15:00 | 2240.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-24 15:15:00 | 2050.00 | 2026-03-25 09:15:00 | 2143.50 | STOP_HIT | 1.00 | -4.56% |
| BUY | retest2 | 2026-03-27 13:15:00 | 2145.90 | 2026-03-30 09:15:00 | 2074.60 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest1 | 2026-04-10 09:15:00 | 2274.40 | 2026-04-13 09:15:00 | 2228.30 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2026-04-13 11:00:00 | 2258.40 | 2026-04-20 15:15:00 | 2279.80 | STOP_HIT | 1.00 | 0.95% |
| BUY | retest2 | 2026-04-13 12:00:00 | 2249.50 | 2026-04-20 15:15:00 | 2279.80 | STOP_HIT | 1.00 | 1.35% |
| BUY | retest2 | 2026-04-13 14:15:00 | 2249.90 | 2026-04-20 15:15:00 | 2279.80 | STOP_HIT | 1.00 | 1.33% |
| BUY | retest2 | 2026-04-15 09:15:00 | 2257.30 | 2026-04-20 15:15:00 | 2279.80 | STOP_HIT | 1.00 | 1.00% |
| SELL | retest2 | 2026-04-28 12:00:00 | 2195.00 | 2026-05-05 09:15:00 | 2204.80 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2026-04-30 09:15:00 | 2175.90 | 2026-05-05 09:15:00 | 2204.80 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2026-05-04 11:30:00 | 2195.20 | 2026-05-05 09:15:00 | 2204.80 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2026-05-04 12:30:00 | 2187.80 | 2026-05-05 09:15:00 | 2204.80 | STOP_HIT | 1.00 | -0.78% |
