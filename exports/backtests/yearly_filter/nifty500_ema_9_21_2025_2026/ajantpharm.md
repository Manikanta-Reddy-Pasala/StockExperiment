# Ajanta Pharmaceuticals Ltd. (AJANTPHARM)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 3033.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 83 |
| ALERT1 | 55 |
| ALERT2 | 55 |
| ALERT2_SKIP | 35 |
| ALERT3 | 131 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 76 |
| PARTIAL | 8 |
| TARGET_HIT | 1 |
| STOP_HIT | 75 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 84 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 34 / 50
- **Target hits / Stop hits / Partials:** 1 / 75 / 8
- **Avg / median % per leg:** 0.54% / -0.24%
- **Sum % (uncompounded):** 45.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 33 | 15 | 45.5% | 1 | 32 | 0 | 0.93% | 30.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 33 | 15 | 45.5% | 1 | 32 | 0 | 0.93% | 30.6% |
| SELL (all) | 51 | 19 | 37.3% | 0 | 43 | 8 | 0.29% | 14.9% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.17% | 0.2% |
| SELL @ 3rd Alert (retest2) | 50 | 18 | 36.0% | 0 | 42 | 8 | 0.29% | 14.7% |
| retest1 (combined) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.17% | 0.2% |
| retest2 (combined) | 83 | 33 | 39.8% | 1 | 74 | 8 | 0.55% | 45.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 2540.20 | 2525.77 | 2525.39 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-12 11:15:00 | 2498.20 | 2523.83 | 2524.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-12 12:15:00 | 2478.40 | 2514.75 | 2520.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-13 09:15:00 | 2523.00 | 2504.75 | 2512.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 09:15:00 | 2523.00 | 2504.75 | 2512.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 2523.00 | 2504.75 | 2512.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-13 10:15:00 | 2530.00 | 2504.75 | 2512.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 10:15:00 | 2559.50 | 2515.70 | 2517.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-13 11:00:00 | 2559.50 | 2515.70 | 2517.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 11:15:00 | 2570.70 | 2526.70 | 2521.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 14:15:00 | 2590.70 | 2552.81 | 2536.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 09:15:00 | 2555.00 | 2558.40 | 2541.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 2555.00 | 2558.40 | 2541.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 2555.00 | 2558.40 | 2541.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 14:45:00 | 2585.80 | 2571.42 | 2554.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 11:45:00 | 2597.80 | 2577.61 | 2563.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:00:00 | 2590.70 | 2580.23 | 2565.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 10:15:00 | 2589.00 | 2588.52 | 2575.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 2659.00 | 2610.23 | 2592.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:30:00 | 2695.50 | 2624.27 | 2618.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 12:15:00 | 2591.60 | 2620.75 | 2622.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 12:15:00 | 2591.60 | 2620.75 | 2622.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 13:15:00 | 2565.70 | 2609.74 | 2617.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 09:15:00 | 2590.40 | 2583.51 | 2594.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 2590.40 | 2583.51 | 2594.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 2590.40 | 2583.51 | 2594.37 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 11:15:00 | 2604.20 | 2593.38 | 2592.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 13:15:00 | 2610.00 | 2598.61 | 2595.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 14:15:00 | 2584.90 | 2595.87 | 2594.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 14:15:00 | 2584.90 | 2595.87 | 2594.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 2584.90 | 2595.87 | 2594.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 15:00:00 | 2584.90 | 2595.87 | 2594.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 15:15:00 | 2580.00 | 2592.69 | 2592.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 14:15:00 | 2575.50 | 2584.48 | 2588.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 13:15:00 | 2550.10 | 2530.12 | 2546.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 13:15:00 | 2550.10 | 2530.12 | 2546.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 2550.10 | 2530.12 | 2546.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:00:00 | 2550.10 | 2530.12 | 2546.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 2550.90 | 2534.27 | 2546.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:30:00 | 2560.80 | 2534.27 | 2546.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 2561.50 | 2542.86 | 2547.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:00:00 | 2561.50 | 2542.86 | 2547.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 11:15:00 | 2560.60 | 2546.41 | 2548.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:30:00 | 2569.80 | 2546.41 | 2548.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 13:15:00 | 2560.10 | 2551.62 | 2551.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 09:15:00 | 2580.40 | 2560.28 | 2555.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 11:15:00 | 2585.80 | 2590.16 | 2576.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-05 11:30:00 | 2582.70 | 2590.16 | 2576.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 12:15:00 | 2564.30 | 2584.99 | 2575.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:45:00 | 2565.00 | 2584.99 | 2575.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 2573.00 | 2582.59 | 2575.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 14:15:00 | 2575.10 | 2582.59 | 2575.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 14:45:00 | 2574.90 | 2581.43 | 2575.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 12:15:00 | 2575.50 | 2581.98 | 2578.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 13:00:00 | 2575.50 | 2580.68 | 2577.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 15:15:00 | 2572.00 | 2576.14 | 2576.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 15:15:00 | 2572.00 | 2576.14 | 2576.23 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 10:15:00 | 2594.00 | 2578.78 | 2577.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 13:15:00 | 2597.10 | 2586.75 | 2581.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 15:15:00 | 2585.10 | 2588.96 | 2583.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 15:15:00 | 2585.10 | 2588.96 | 2583.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 2585.10 | 2588.96 | 2583.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:45:00 | 2582.80 | 2587.85 | 2583.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 2576.60 | 2585.60 | 2583.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:30:00 | 2573.50 | 2585.60 | 2583.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 2574.50 | 2583.38 | 2582.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:30:00 | 2571.40 | 2583.38 | 2582.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 13:15:00 | 2565.20 | 2578.63 | 2580.25 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 10:15:00 | 2639.30 | 2591.10 | 2585.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 14:15:00 | 2647.90 | 2621.99 | 2603.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 2678.00 | 2702.66 | 2668.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 2678.00 | 2702.66 | 2668.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 2678.00 | 2702.66 | 2668.53 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2025-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 15:15:00 | 2632.00 | 2655.05 | 2655.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 2616.80 | 2647.40 | 2652.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 10:15:00 | 2570.70 | 2563.15 | 2588.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-18 11:00:00 | 2570.70 | 2563.15 | 2588.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 2584.80 | 2568.96 | 2583.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 15:00:00 | 2584.80 | 2568.96 | 2583.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 2575.50 | 2570.27 | 2582.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 2584.70 | 2570.27 | 2582.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 2558.50 | 2567.91 | 2580.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:15:00 | 2547.80 | 2567.91 | 2580.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:30:00 | 2551.50 | 2562.94 | 2575.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 11:00:00 | 2550.80 | 2552.80 | 2563.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 15:15:00 | 2590.00 | 2568.29 | 2567.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 15:15:00 | 2590.00 | 2568.29 | 2567.34 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 10:15:00 | 2565.10 | 2566.54 | 2566.64 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 2575.00 | 2566.47 | 2566.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 11:15:00 | 2587.80 | 2572.45 | 2569.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 2579.60 | 2582.93 | 2576.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 09:15:00 | 2579.60 | 2582.93 | 2576.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 2579.60 | 2582.93 | 2576.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:00:00 | 2579.60 | 2582.93 | 2576.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 2582.20 | 2582.78 | 2577.02 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 15:15:00 | 2573.00 | 2574.36 | 2574.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 10:15:00 | 2561.50 | 2570.55 | 2572.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 11:15:00 | 2573.70 | 2571.18 | 2572.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 11:15:00 | 2573.70 | 2571.18 | 2572.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 2573.70 | 2571.18 | 2572.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 12:00:00 | 2573.70 | 2571.18 | 2572.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 2553.80 | 2567.70 | 2571.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 13:15:00 | 2547.50 | 2567.70 | 2571.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 14:45:00 | 2535.70 | 2556.31 | 2565.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 12:00:00 | 2547.20 | 2556.39 | 2562.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 15:15:00 | 2575.00 | 2565.31 | 2565.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-06-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 15:15:00 | 2575.00 | 2565.31 | 2565.03 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 2529.50 | 2559.79 | 2562.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 11:15:00 | 2508.60 | 2549.55 | 2557.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 11:15:00 | 2519.50 | 2516.38 | 2532.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-02 12:00:00 | 2519.50 | 2516.38 | 2532.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 2555.00 | 2524.10 | 2534.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:45:00 | 2550.70 | 2524.10 | 2534.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 2560.00 | 2531.28 | 2536.99 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 09:15:00 | 2565.30 | 2544.08 | 2541.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 11:15:00 | 2621.00 | 2563.32 | 2551.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 10:15:00 | 2665.90 | 2668.21 | 2634.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-07 10:45:00 | 2664.50 | 2668.21 | 2634.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 2633.40 | 2654.40 | 2641.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 2633.40 | 2654.40 | 2641.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 2607.10 | 2644.94 | 2638.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:00:00 | 2607.10 | 2644.94 | 2638.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 2616.50 | 2639.25 | 2636.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:45:00 | 2609.10 | 2639.25 | 2636.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 13:15:00 | 2622.00 | 2633.12 | 2634.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 14:15:00 | 2610.50 | 2628.60 | 2631.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 10:15:00 | 2624.70 | 2623.73 | 2628.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 10:15:00 | 2624.70 | 2623.73 | 2628.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 2624.70 | 2623.73 | 2628.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 11:15:00 | 2618.50 | 2623.73 | 2628.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:30:00 | 2616.00 | 2619.26 | 2623.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:45:00 | 2613.50 | 2605.87 | 2611.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 11:15:00 | 2634.80 | 2611.66 | 2613.56 | SL hit (close>static) qty=1.00 sl=2634.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 12:15:00 | 2634.00 | 2616.12 | 2615.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 12:15:00 | 2671.10 | 2634.14 | 2625.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 14:15:00 | 2781.40 | 2785.64 | 2748.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 15:00:00 | 2781.40 | 2785.64 | 2748.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 2776.90 | 2783.93 | 2762.16 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 11:15:00 | 2745.00 | 2763.85 | 2764.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 15:15:00 | 2735.00 | 2751.38 | 2757.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 2769.10 | 2736.30 | 2743.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 2769.10 | 2736.30 | 2743.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 2769.10 | 2736.30 | 2743.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:00:00 | 2769.10 | 2736.30 | 2743.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 2804.70 | 2749.98 | 2749.51 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 12:15:00 | 2739.00 | 2772.96 | 2774.10 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 12:15:00 | 2812.40 | 2776.41 | 2773.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 13:15:00 | 2824.50 | 2786.03 | 2778.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 09:15:00 | 2799.20 | 2799.75 | 2787.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 09:15:00 | 2799.20 | 2799.75 | 2787.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 2799.20 | 2799.75 | 2787.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:45:00 | 2789.00 | 2799.75 | 2787.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 2792.40 | 2800.91 | 2794.72 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 11:15:00 | 2746.70 | 2786.61 | 2789.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 15:15:00 | 2742.00 | 2763.67 | 2776.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 12:15:00 | 2574.50 | 2566.68 | 2597.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 13:00:00 | 2574.50 | 2566.68 | 2597.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 2599.30 | 2573.21 | 2597.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:00:00 | 2599.30 | 2573.21 | 2597.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 2619.70 | 2582.50 | 2599.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 2619.70 | 2582.50 | 2599.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 2612.00 | 2588.40 | 2600.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 2589.20 | 2588.40 | 2600.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 2599.40 | 2594.55 | 2601.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 11:15:00 | 2594.40 | 2594.55 | 2601.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 13:15:00 | 2592.60 | 2598.09 | 2602.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 13:45:00 | 2591.80 | 2597.18 | 2601.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 10:15:00 | 2618.00 | 2605.35 | 2604.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 10:15:00 | 2618.00 | 2605.35 | 2604.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 11:15:00 | 2640.40 | 2612.36 | 2607.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 2673.60 | 2674.66 | 2653.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 09:45:00 | 2677.00 | 2674.66 | 2653.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 2666.10 | 2674.72 | 2657.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:45:00 | 2661.20 | 2674.72 | 2657.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 2700.30 | 2679.81 | 2666.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 10:45:00 | 2716.50 | 2682.84 | 2673.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 12:15:00 | 2655.00 | 2682.32 | 2680.00 | SL hit (close<static) qty=1.00 sl=2660.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 13:15:00 | 2637.30 | 2673.32 | 2676.12 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 14:15:00 | 2702.30 | 2679.11 | 2678.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 15:15:00 | 2737.00 | 2690.69 | 2683.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 2671.80 | 2686.91 | 2682.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 09:15:00 | 2671.80 | 2686.91 | 2682.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 2671.80 | 2686.91 | 2682.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:00:00 | 2671.80 | 2686.91 | 2682.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 2691.50 | 2687.83 | 2683.52 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 09:15:00 | 2671.80 | 2681.14 | 2681.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 14:15:00 | 2654.70 | 2669.71 | 2675.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 15:15:00 | 2499.80 | 2491.79 | 2524.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 15:15:00 | 2499.80 | 2491.79 | 2524.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 2499.80 | 2491.79 | 2524.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 10:15:00 | 2449.60 | 2485.71 | 2518.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 12:30:00 | 2450.10 | 2470.91 | 2502.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 14:45:00 | 2450.50 | 2465.46 | 2494.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 15:15:00 | 2448.70 | 2465.46 | 2494.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 2490.00 | 2470.77 | 2483.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 2490.00 | 2470.77 | 2483.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 2480.00 | 2472.61 | 2483.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 15:15:00 | 2475.00 | 2472.61 | 2483.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 10:15:00 | 2451.00 | 2474.61 | 2482.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 11:15:00 | 2529.30 | 2487.48 | 2487.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 11:15:00 | 2529.30 | 2487.48 | 2487.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 13:15:00 | 2559.80 | 2508.85 | 2497.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 09:15:00 | 2602.10 | 2605.54 | 2580.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 2602.10 | 2605.54 | 2580.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 2602.10 | 2605.54 | 2580.51 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 13:15:00 | 2564.60 | 2578.27 | 2579.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 15:15:00 | 2555.00 | 2570.70 | 2576.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 2581.30 | 2572.82 | 2576.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 2581.30 | 2572.82 | 2576.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 2581.30 | 2572.82 | 2576.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:00:00 | 2581.30 | 2572.82 | 2576.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 2579.50 | 2574.15 | 2576.76 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 12:15:00 | 2592.30 | 2579.77 | 2578.99 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 10:15:00 | 2567.20 | 2578.26 | 2579.21 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 10:15:00 | 2586.80 | 2579.48 | 2578.94 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 11:15:00 | 2572.20 | 2578.02 | 2578.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 09:15:00 | 2568.00 | 2574.85 | 2576.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 13:15:00 | 2548.00 | 2546.48 | 2556.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 14:00:00 | 2548.00 | 2546.48 | 2556.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 2553.40 | 2547.86 | 2556.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:45:00 | 2553.70 | 2547.86 | 2556.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 2549.80 | 2548.25 | 2555.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:30:00 | 2551.40 | 2549.88 | 2555.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 2548.90 | 2549.68 | 2555.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 11:15:00 | 2547.10 | 2549.68 | 2555.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 14:15:00 | 2562.10 | 2553.74 | 2555.50 | SL hit (close>static) qty=1.00 sl=2559.20 alert=retest2 |

### Cycle 37 — BUY (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 14:15:00 | 2430.00 | 2415.52 | 2414.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 12:15:00 | 2453.20 | 2428.81 | 2422.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 2424.00 | 2438.61 | 2430.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 2424.00 | 2438.61 | 2430.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 2424.00 | 2438.61 | 2430.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 2424.00 | 2438.61 | 2430.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 2424.20 | 2435.73 | 2429.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:45:00 | 2425.00 | 2435.73 | 2429.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 12:15:00 | 2400.20 | 2422.91 | 2424.41 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 2448.30 | 2424.16 | 2422.21 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 2402.80 | 2423.19 | 2423.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 11:15:00 | 2402.30 | 2419.01 | 2421.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 14:15:00 | 2419.00 | 2415.32 | 2418.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 14:15:00 | 2419.00 | 2415.32 | 2418.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 2419.00 | 2415.32 | 2418.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:45:00 | 2420.00 | 2415.32 | 2418.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 2405.00 | 2413.26 | 2417.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 09:15:00 | 2397.90 | 2413.26 | 2417.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 2429.80 | 2403.23 | 2407.63 | SL hit (close>static) qty=1.00 sl=2421.90 alert=retest2 |

### Cycle 41 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 2460.10 | 2416.76 | 2412.17 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 12:15:00 | 2409.30 | 2427.77 | 2429.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 12:15:00 | 2398.20 | 2412.05 | 2419.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 14:15:00 | 2414.00 | 2410.67 | 2417.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 14:15:00 | 2414.00 | 2410.67 | 2417.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 2414.00 | 2410.67 | 2417.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 15:00:00 | 2414.00 | 2410.67 | 2417.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 2414.90 | 2411.52 | 2417.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:15:00 | 2432.80 | 2411.52 | 2417.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 2419.00 | 2413.01 | 2417.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:45:00 | 2430.00 | 2413.01 | 2417.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 2412.00 | 2412.81 | 2417.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 12:15:00 | 2398.40 | 2411.39 | 2416.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 11:45:00 | 2397.50 | 2403.35 | 2408.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 2435.10 | 2406.28 | 2406.42 | SL hit (close>static) qty=1.00 sl=2421.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 2432.70 | 2411.57 | 2408.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 12:15:00 | 2449.80 | 2419.21 | 2412.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 2460.20 | 2461.72 | 2447.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 11:00:00 | 2460.20 | 2461.72 | 2447.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 2530.20 | 2481.58 | 2466.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 15:00:00 | 2542.90 | 2493.84 | 2473.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 09:30:00 | 2543.70 | 2515.29 | 2487.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 13:15:00 | 2523.10 | 2539.65 | 2541.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 13:15:00 | 2523.10 | 2539.65 | 2541.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 09:15:00 | 2509.60 | 2528.12 | 2535.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 12:15:00 | 2525.80 | 2525.26 | 2531.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 13:00:00 | 2525.80 | 2525.26 | 2531.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 2524.10 | 2524.89 | 2530.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 15:00:00 | 2524.10 | 2524.89 | 2530.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 2524.10 | 2524.73 | 2530.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 2516.80 | 2524.73 | 2530.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 10:15:00 | 2562.20 | 2535.61 | 2534.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 10:15:00 | 2562.20 | 2535.61 | 2534.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 13:15:00 | 2575.80 | 2550.58 | 2542.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 15:15:00 | 2560.00 | 2568.63 | 2559.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 15:15:00 | 2560.00 | 2568.63 | 2559.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 2560.00 | 2568.63 | 2559.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:15:00 | 2553.20 | 2568.63 | 2559.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 2554.10 | 2565.72 | 2559.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:45:00 | 2556.00 | 2565.72 | 2559.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 2566.00 | 2565.78 | 2559.90 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 14:15:00 | 2550.10 | 2556.51 | 2556.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 2530.00 | 2549.84 | 2553.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 15:15:00 | 2530.00 | 2527.59 | 2538.46 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-17 09:15:00 | 2500.00 | 2527.59 | 2538.46 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 2504.40 | 2506.07 | 2521.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:30:00 | 2512.10 | 2506.07 | 2521.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 2484.30 | 2477.19 | 2491.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:45:00 | 2493.00 | 2477.19 | 2491.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 2495.70 | 2480.89 | 2492.33 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-19 11:15:00 | 2495.70 | 2480.89 | 2492.33 | SL hit (close>ema400) qty=1.00 sl=2492.33 alert=retest1 |

### Cycle 47 — BUY (started 2025-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 15:15:00 | 2511.00 | 2499.24 | 2498.46 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 2473.20 | 2502.56 | 2502.69 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 13:15:00 | 2500.90 | 2495.47 | 2495.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 14:15:00 | 2521.20 | 2500.62 | 2497.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 15:15:00 | 2500.00 | 2500.49 | 2497.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-25 09:15:00 | 2491.60 | 2500.49 | 2497.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 2500.00 | 2500.39 | 2498.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 11:15:00 | 2517.60 | 2500.71 | 2498.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 13:45:00 | 2517.30 | 2506.99 | 2502.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 14:15:00 | 2513.30 | 2506.99 | 2502.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 09:15:00 | 2519.70 | 2506.84 | 2503.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 2517.30 | 2508.93 | 2504.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 10:15:00 | 2533.30 | 2508.93 | 2504.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 11:30:00 | 2540.10 | 2516.64 | 2508.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 13:15:00 | 2530.00 | 2518.53 | 2510.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 14:15:00 | 2530.70 | 2519.18 | 2511.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 2508.50 | 2521.78 | 2514.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:45:00 | 2508.40 | 2521.78 | 2514.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 2508.20 | 2519.06 | 2514.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 12:15:00 | 2525.20 | 2517.87 | 2514.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 2621.40 | 2635.23 | 2635.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 12:15:00 | 2621.40 | 2635.23 | 2635.46 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 2640.20 | 2634.93 | 2634.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 12:15:00 | 2646.00 | 2637.14 | 2635.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 09:15:00 | 2625.90 | 2641.97 | 2638.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 2625.90 | 2641.97 | 2638.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 2625.90 | 2641.97 | 2638.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:00:00 | 2625.90 | 2641.97 | 2638.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 2632.40 | 2640.06 | 2638.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:45:00 | 2624.90 | 2640.06 | 2638.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 2639.00 | 2639.85 | 2638.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:45:00 | 2623.40 | 2639.85 | 2638.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 2643.30 | 2640.54 | 2638.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:00:00 | 2643.30 | 2640.54 | 2638.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 2639.20 | 2640.27 | 2638.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:00:00 | 2639.20 | 2640.27 | 2638.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 2646.40 | 2641.50 | 2639.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:45:00 | 2638.00 | 2641.50 | 2639.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 2670.00 | 2647.20 | 2642.34 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2025-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 13:15:00 | 2636.20 | 2643.80 | 2644.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 2619.30 | 2638.28 | 2641.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 12:15:00 | 2601.80 | 2597.47 | 2612.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 12:30:00 | 2599.00 | 2597.47 | 2612.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 2570.80 | 2591.18 | 2604.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 11:15:00 | 2565.00 | 2586.92 | 2601.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 12:30:00 | 2564.50 | 2575.86 | 2593.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 2670.00 | 2602.14 | 2600.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 2670.00 | 2602.14 | 2600.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 11:15:00 | 2685.10 | 2647.10 | 2627.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 10:15:00 | 2675.00 | 2678.03 | 2654.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 11:00:00 | 2675.00 | 2678.03 | 2654.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 2651.00 | 2672.62 | 2654.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:45:00 | 2655.00 | 2672.62 | 2654.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 2663.40 | 2670.78 | 2655.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 2766.50 | 2667.80 | 2657.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 2687.00 | 2698.82 | 2698.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 2685.50 | 2696.16 | 2697.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 2685.50 | 2696.16 | 2697.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 12:15:00 | 2671.00 | 2686.86 | 2692.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 13:15:00 | 2693.50 | 2688.19 | 2692.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 13:15:00 | 2693.50 | 2688.19 | 2692.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 2693.50 | 2688.19 | 2692.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 13:45:00 | 2694.30 | 2688.19 | 2692.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 2690.30 | 2688.61 | 2692.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:30:00 | 2694.60 | 2688.61 | 2692.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 2694.00 | 2689.69 | 2692.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 2718.20 | 2689.69 | 2692.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 09:15:00 | 2728.30 | 2697.41 | 2695.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 10:15:00 | 2751.00 | 2708.13 | 2700.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 15:15:00 | 2720.00 | 2721.75 | 2711.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:15:00 | 2726.20 | 2721.75 | 2711.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 2724.40 | 2722.28 | 2712.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 10:15:00 | 2745.60 | 2722.28 | 2712.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-02 12:15:00 | 3020.16 | 2921.92 | 2856.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 2816.80 | 2887.72 | 2893.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 12:15:00 | 2788.90 | 2822.87 | 2840.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 2750.00 | 2742.49 | 2775.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 2724.70 | 2742.49 | 2775.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 2681.30 | 2699.30 | 2718.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:00:00 | 2665.70 | 2686.73 | 2707.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 2664.40 | 2683.56 | 2700.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 15:15:00 | 2669.00 | 2694.58 | 2699.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 11:00:00 | 2674.90 | 2686.14 | 2694.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 13:15:00 | 2692.50 | 2684.08 | 2690.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 14:00:00 | 2692.50 | 2684.08 | 2690.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 2681.10 | 2683.48 | 2690.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 14:30:00 | 2699.70 | 2683.48 | 2690.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 2695.00 | 2685.79 | 2690.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:15:00 | 2689.00 | 2685.79 | 2690.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 2705.40 | 2689.71 | 2691.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:45:00 | 2701.00 | 2689.71 | 2691.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 2675.10 | 2686.79 | 2690.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-21 12:15:00 | 2702.80 | 2693.11 | 2692.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 12:15:00 | 2702.80 | 2693.11 | 2692.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 09:15:00 | 2740.70 | 2708.58 | 2700.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-22 14:15:00 | 2717.40 | 2717.80 | 2709.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-22 15:00:00 | 2717.40 | 2717.80 | 2709.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 2699.50 | 2713.53 | 2708.66 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 2692.60 | 2703.72 | 2705.18 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 09:15:00 | 2718.60 | 2705.63 | 2705.58 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-01-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 12:15:00 | 2703.20 | 2705.48 | 2705.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 2700.10 | 2704.40 | 2705.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 10:15:00 | 2709.00 | 2702.52 | 2703.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 10:15:00 | 2709.00 | 2702.52 | 2703.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 2709.00 | 2702.52 | 2703.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 2709.00 | 2702.52 | 2703.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 2701.90 | 2702.39 | 2703.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 12:15:00 | 2700.10 | 2702.39 | 2703.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 14:15:00 | 2719.50 | 2704.93 | 2704.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 2719.50 | 2704.93 | 2704.28 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 13:15:00 | 2678.00 | 2699.51 | 2702.14 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 2766.40 | 2711.84 | 2707.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 13:15:00 | 2805.00 | 2777.99 | 2753.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 10:15:00 | 2781.60 | 2789.64 | 2768.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 10:15:00 | 2781.60 | 2789.64 | 2768.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 2781.60 | 2789.64 | 2768.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:30:00 | 2770.40 | 2789.64 | 2768.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 2772.80 | 2786.92 | 2770.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:00:00 | 2772.80 | 2786.92 | 2770.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 2776.30 | 2784.80 | 2771.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 14:30:00 | 2789.50 | 2784.28 | 2772.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 10:15:00 | 2785.00 | 2812.64 | 2813.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 2785.00 | 2812.64 | 2813.44 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 12:15:00 | 2820.30 | 2810.74 | 2810.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 14:15:00 | 2854.90 | 2820.89 | 2815.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 12:15:00 | 2899.40 | 2900.67 | 2887.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 14:15:00 | 2891.80 | 2897.96 | 2888.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 2891.80 | 2897.96 | 2888.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:30:00 | 2888.00 | 2897.96 | 2888.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 2899.90 | 2898.34 | 2889.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:15:00 | 2876.00 | 2898.34 | 2889.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 2906.10 | 2899.90 | 2891.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 10:30:00 | 2912.80 | 2904.54 | 2894.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 14:15:00 | 2915.20 | 2911.87 | 2900.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:30:00 | 2949.20 | 2956.32 | 2956.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 10:15:00 | 2944.40 | 2953.94 | 2955.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 10:15:00 | 2944.40 | 2953.94 | 2955.08 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 2970.00 | 2953.26 | 2952.48 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 12:15:00 | 2948.00 | 2957.46 | 2957.71 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 2967.70 | 2957.49 | 2957.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 11:15:00 | 3020.00 | 2972.72 | 2964.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 12:15:00 | 3002.00 | 3015.95 | 2996.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 13:00:00 | 3002.00 | 3015.95 | 2996.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 2979.90 | 3008.74 | 2994.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 14:00:00 | 2979.90 | 3008.74 | 2994.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 2975.00 | 3002.00 | 2993.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 15:15:00 | 2994.90 | 3002.00 | 2993.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 13:30:00 | 2981.30 | 3004.14 | 2998.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 2938.00 | 2985.27 | 2990.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 2938.00 | 2985.27 | 2990.97 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 15:15:00 | 2975.00 | 2950.28 | 2949.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 2996.00 | 2959.42 | 2953.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 2958.50 | 2981.22 | 2971.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 2958.50 | 2981.22 | 2971.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 2958.50 | 2981.22 | 2971.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 11:30:00 | 2963.00 | 2969.87 | 2967.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 12:15:00 | 2950.90 | 2966.08 | 2966.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2026-03-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 12:15:00 | 2950.90 | 2966.08 | 2966.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 13:15:00 | 2944.10 | 2961.68 | 2964.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 2993.30 | 2962.61 | 2963.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 2993.30 | 2962.61 | 2963.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 2993.30 | 2962.61 | 2963.34 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 3000.70 | 2970.23 | 2966.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 3012.30 | 2992.92 | 2980.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 3036.90 | 3088.09 | 3062.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 3036.90 | 3088.09 | 3062.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 3036.90 | 3088.09 | 3062.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 09:45:00 | 3044.50 | 3088.09 | 3062.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 3026.90 | 3075.85 | 3059.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 12:45:00 | 3040.90 | 3061.42 | 3055.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 14:00:00 | 3038.00 | 3056.74 | 3053.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 15:15:00 | 3030.00 | 3047.11 | 3049.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 15:15:00 | 3030.00 | 3047.11 | 3049.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 2991.30 | 3035.95 | 3044.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 3054.50 | 2988.48 | 3007.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 3054.50 | 2988.48 | 3007.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 3054.50 | 2988.48 | 3007.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 3054.50 | 2988.48 | 3007.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 3059.40 | 3002.66 | 3011.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:30:00 | 3063.70 | 3002.66 | 3011.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 3024.40 | 3013.58 | 3015.13 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 15:15:00 | 3024.90 | 3017.00 | 3016.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 3044.00 | 3022.40 | 3018.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 3040.20 | 3052.79 | 3041.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 3040.20 | 3052.79 | 3041.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 3040.20 | 3052.79 | 3041.07 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 3001.00 | 3033.47 | 3033.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 2940.80 | 3008.38 | 3021.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 15:15:00 | 3020.00 | 2937.91 | 2963.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 15:15:00 | 3020.00 | 2937.91 | 2963.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 3020.00 | 2937.91 | 2963.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 2854.30 | 2937.91 | 2963.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 11:15:00 | 2925.10 | 2882.33 | 2879.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 2925.10 | 2882.33 | 2879.28 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 2840.90 | 2876.01 | 2878.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 2804.20 | 2845.11 | 2860.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 2860.20 | 2816.19 | 2830.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 2860.20 | 2816.19 | 2830.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 2860.20 | 2816.19 | 2830.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:45:00 | 2829.30 | 2820.05 | 2830.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:15:00 | 2818.20 | 2824.58 | 2831.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-09 14:15:00 | 2756.20 | 2740.72 | 2739.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2026-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-09 14:15:00 | 2756.20 | 2740.72 | 2739.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 09:15:00 | 2808.30 | 2755.72 | 2746.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-15 13:15:00 | 2883.00 | 2886.80 | 2855.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-15 14:00:00 | 2883.00 | 2886.80 | 2855.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 2830.20 | 2874.26 | 2857.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 09:45:00 | 2824.60 | 2874.26 | 2857.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 2824.30 | 2864.27 | 2854.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 10:45:00 | 2823.50 | 2864.27 | 2854.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2026-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 12:15:00 | 2822.70 | 2848.68 | 2848.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-17 13:15:00 | 2807.70 | 2827.93 | 2836.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 09:15:00 | 2798.20 | 2793.43 | 2808.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 09:15:00 | 2798.20 | 2793.43 | 2808.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 2798.20 | 2793.43 | 2808.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 09:45:00 | 2800.90 | 2793.43 | 2808.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 2793.10 | 2791.51 | 2803.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 12:30:00 | 2798.90 | 2791.51 | 2803.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 2797.60 | 2791.76 | 2800.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:15:00 | 2791.10 | 2791.76 | 2800.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 2755.00 | 2784.41 | 2796.37 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 2803.60 | 2784.53 | 2784.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 13:15:00 | 2815.00 | 2796.69 | 2790.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 09:15:00 | 2783.00 | 2798.14 | 2793.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 09:15:00 | 2783.00 | 2798.14 | 2793.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 2783.00 | 2798.14 | 2793.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:30:00 | 2783.00 | 2798.14 | 2793.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 2774.60 | 2793.43 | 2791.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:00:00 | 2774.60 | 2793.43 | 2791.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 11:15:00 | 2761.20 | 2786.98 | 2788.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 12:15:00 | 2752.20 | 2780.03 | 2785.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 2805.00 | 2775.70 | 2780.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 09:15:00 | 2805.00 | 2775.70 | 2780.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 2805.00 | 2775.70 | 2780.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:45:00 | 2816.50 | 2775.70 | 2780.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 10:15:00 | 2815.60 | 2783.68 | 2783.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 12:15:00 | 2833.30 | 2808.69 | 2798.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 2883.60 | 2884.24 | 2855.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 10:00:00 | 2883.60 | 2884.24 | 2855.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 2900.00 | 2892.53 | 2872.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 2975.70 | 2892.53 | 2872.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 14:45:00 | 2585.80 | 2025-05-22 12:15:00 | 2591.60 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2025-05-15 11:45:00 | 2597.80 | 2025-05-22 12:15:00 | 2591.60 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-05-15 13:00:00 | 2590.70 | 2025-05-22 12:15:00 | 2591.60 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2025-05-16 10:15:00 | 2589.00 | 2025-05-22 12:15:00 | 2591.60 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2025-05-21 09:30:00 | 2695.50 | 2025-05-22 12:15:00 | 2591.60 | STOP_HIT | 1.00 | -3.85% |
| BUY | retest2 | 2025-06-05 14:15:00 | 2575.10 | 2025-06-06 15:15:00 | 2572.00 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2025-06-05 14:45:00 | 2574.90 | 2025-06-06 15:15:00 | 2572.00 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-06-06 12:15:00 | 2575.50 | 2025-06-06 15:15:00 | 2572.00 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2025-06-06 13:00:00 | 2575.50 | 2025-06-06 15:15:00 | 2572.00 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2025-06-19 10:15:00 | 2547.80 | 2025-06-20 15:15:00 | 2590.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-06-19 11:30:00 | 2551.50 | 2025-06-20 15:15:00 | 2590.00 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-06-20 11:00:00 | 2550.80 | 2025-06-20 15:15:00 | 2590.00 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-06-27 13:15:00 | 2547.50 | 2025-06-30 15:15:00 | 2575.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-06-27 14:45:00 | 2535.70 | 2025-06-30 15:15:00 | 2575.00 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-06-30 12:00:00 | 2547.20 | 2025-06-30 15:15:00 | 2575.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-07-09 11:15:00 | 2618.50 | 2025-07-11 11:15:00 | 2634.80 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-07-10 09:30:00 | 2616.00 | 2025-07-11 11:15:00 | 2634.80 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-07-11 10:45:00 | 2613.50 | 2025-07-11 11:15:00 | 2634.80 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-08-08 11:15:00 | 2594.40 | 2025-08-11 10:15:00 | 2618.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-08-08 13:15:00 | 2592.60 | 2025-08-11 10:15:00 | 2618.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-08-08 13:45:00 | 2591.80 | 2025-08-11 10:15:00 | 2618.00 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-08-18 10:45:00 | 2716.50 | 2025-08-19 12:15:00 | 2655.00 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-09-01 10:15:00 | 2449.60 | 2025-09-03 11:15:00 | 2529.30 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2025-09-01 12:30:00 | 2450.10 | 2025-09-03 11:15:00 | 2529.30 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2025-09-01 14:45:00 | 2450.50 | 2025-09-03 11:15:00 | 2529.30 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2025-09-01 15:15:00 | 2448.70 | 2025-09-03 11:15:00 | 2529.30 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2025-09-02 15:15:00 | 2475.00 | 2025-09-03 11:15:00 | 2529.30 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-09-03 10:15:00 | 2451.00 | 2025-09-03 11:15:00 | 2529.30 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2025-09-17 11:15:00 | 2547.10 | 2025-09-17 14:15:00 | 2562.10 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-09-18 12:30:00 | 2547.00 | 2025-09-26 09:15:00 | 2419.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 15:15:00 | 2546.50 | 2025-09-26 09:15:00 | 2419.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 10:45:00 | 2544.00 | 2025-09-26 09:15:00 | 2416.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 09:15:00 | 2519.20 | 2025-09-26 09:15:00 | 2400.55 | PARTIAL | 0.50 | 4.71% |
| SELL | retest2 | 2025-09-18 12:30:00 | 2547.00 | 2025-09-30 10:15:00 | 2455.00 | STOP_HIT | 0.50 | 3.61% |
| SELL | retest2 | 2025-09-18 15:15:00 | 2546.50 | 2025-09-30 10:15:00 | 2455.00 | STOP_HIT | 0.50 | 3.59% |
| SELL | retest2 | 2025-09-19 10:45:00 | 2544.00 | 2025-09-30 10:15:00 | 2455.00 | STOP_HIT | 0.50 | 3.50% |
| SELL | retest2 | 2025-09-24 09:15:00 | 2519.20 | 2025-09-30 10:15:00 | 2455.00 | STOP_HIT | 0.50 | 2.55% |
| SELL | retest2 | 2025-09-24 13:45:00 | 2524.00 | 2025-09-30 13:15:00 | 2393.24 | PARTIAL | 0.50 | 5.18% |
| SELL | retest2 | 2025-09-24 14:45:00 | 2526.90 | 2025-09-30 13:15:00 | 2397.80 | PARTIAL | 0.50 | 5.11% |
| SELL | retest2 | 2025-09-24 15:15:00 | 2520.00 | 2025-09-30 13:15:00 | 2394.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 10:30:00 | 2508.00 | 2025-09-30 13:15:00 | 2382.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 13:45:00 | 2524.00 | 2025-10-01 09:15:00 | 2428.40 | STOP_HIT | 0.50 | 3.79% |
| SELL | retest2 | 2025-09-24 14:45:00 | 2526.90 | 2025-10-01 09:15:00 | 2428.40 | STOP_HIT | 0.50 | 3.90% |
| SELL | retest2 | 2025-09-24 15:15:00 | 2520.00 | 2025-10-01 09:15:00 | 2428.40 | STOP_HIT | 0.50 | 3.63% |
| SELL | retest2 | 2025-09-25 10:30:00 | 2508.00 | 2025-10-01 09:15:00 | 2428.40 | STOP_HIT | 0.50 | 3.17% |
| SELL | retest2 | 2025-10-14 09:15:00 | 2397.90 | 2025-10-15 09:15:00 | 2429.80 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-10-15 12:15:00 | 2398.50 | 2025-10-16 09:15:00 | 2460.10 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-10-27 12:15:00 | 2398.40 | 2025-10-29 10:15:00 | 2435.10 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-10-28 11:45:00 | 2397.50 | 2025-10-29 10:15:00 | 2435.10 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-11-03 15:00:00 | 2542.90 | 2025-11-07 13:15:00 | 2523.10 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-11-04 09:30:00 | 2543.70 | 2025-11-07 13:15:00 | 2523.10 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-11-11 09:15:00 | 2516.80 | 2025-11-11 10:15:00 | 2562.20 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest1 | 2025-11-17 09:15:00 | 2500.00 | 2025-11-19 11:15:00 | 2495.70 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2025-11-25 11:15:00 | 2517.60 | 2025-12-09 12:15:00 | 2621.40 | STOP_HIT | 1.00 | 4.12% |
| BUY | retest2 | 2025-11-25 13:45:00 | 2517.30 | 2025-12-09 12:15:00 | 2621.40 | STOP_HIT | 1.00 | 4.14% |
| BUY | retest2 | 2025-11-25 14:15:00 | 2513.30 | 2025-12-09 12:15:00 | 2621.40 | STOP_HIT | 1.00 | 4.30% |
| BUY | retest2 | 2025-11-26 09:15:00 | 2519.70 | 2025-12-09 12:15:00 | 2621.40 | STOP_HIT | 1.00 | 4.04% |
| BUY | retest2 | 2025-11-26 10:15:00 | 2533.30 | 2025-12-09 12:15:00 | 2621.40 | STOP_HIT | 1.00 | 3.48% |
| BUY | retest2 | 2025-11-26 11:30:00 | 2540.10 | 2025-12-09 12:15:00 | 2621.40 | STOP_HIT | 1.00 | 3.20% |
| BUY | retest2 | 2025-11-26 13:15:00 | 2530.00 | 2025-12-09 12:15:00 | 2621.40 | STOP_HIT | 1.00 | 3.61% |
| BUY | retest2 | 2025-11-26 14:15:00 | 2530.70 | 2025-12-09 12:15:00 | 2621.40 | STOP_HIT | 1.00 | 3.58% |
| BUY | retest2 | 2025-11-27 12:15:00 | 2525.20 | 2025-12-09 12:15:00 | 2621.40 | STOP_HIT | 1.00 | 3.81% |
| SELL | retest2 | 2025-12-18 11:15:00 | 2565.00 | 2025-12-19 09:15:00 | 2670.00 | STOP_HIT | 1.00 | -4.09% |
| SELL | retest2 | 2025-12-18 12:30:00 | 2564.50 | 2025-12-19 09:15:00 | 2670.00 | STOP_HIT | 1.00 | -4.11% |
| BUY | retest2 | 2025-12-24 09:15:00 | 2766.50 | 2025-12-29 09:15:00 | 2685.50 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2025-12-29 09:15:00 | 2687.00 | 2025-12-29 09:15:00 | 2685.50 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2025-12-31 10:15:00 | 2745.60 | 2026-01-02 12:15:00 | 3020.16 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-16 13:00:00 | 2665.70 | 2026-01-21 12:15:00 | 2702.80 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-01-19 09:15:00 | 2664.40 | 2026-01-21 12:15:00 | 2702.80 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2026-01-19 15:15:00 | 2669.00 | 2026-01-21 12:15:00 | 2702.80 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2026-01-20 11:00:00 | 2674.90 | 2026-01-21 12:15:00 | 2702.80 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2026-01-28 12:15:00 | 2700.10 | 2026-01-28 14:15:00 | 2719.50 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2026-02-02 14:30:00 | 2789.50 | 2026-02-05 10:15:00 | 2785.00 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2026-02-13 10:30:00 | 2912.80 | 2026-02-20 10:15:00 | 2944.40 | STOP_HIT | 1.00 | 1.08% |
| BUY | retest2 | 2026-02-13 14:15:00 | 2915.20 | 2026-02-20 10:15:00 | 2944.40 | STOP_HIT | 1.00 | 1.00% |
| BUY | retest2 | 2026-02-20 09:30:00 | 2949.20 | 2026-02-20 10:15:00 | 2944.40 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2026-02-26 15:15:00 | 2994.90 | 2026-03-02 09:15:00 | 2938.00 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2026-02-27 13:30:00 | 2981.30 | 2026-03-02 09:15:00 | 2938.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2026-03-09 11:30:00 | 2963.00 | 2026-03-09 12:15:00 | 2950.90 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2026-03-13 12:45:00 | 3040.90 | 2026-03-13 15:15:00 | 3030.00 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2026-03-13 14:00:00 | 3038.00 | 2026-03-13 15:15:00 | 3030.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2026-03-23 09:15:00 | 2854.30 | 2026-03-25 11:15:00 | 2925.10 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2026-04-01 10:45:00 | 2829.30 | 2026-04-09 14:15:00 | 2756.20 | STOP_HIT | 1.00 | 2.58% |
| SELL | retest2 | 2026-04-01 13:15:00 | 2818.20 | 2026-04-09 14:15:00 | 2756.20 | STOP_HIT | 1.00 | 2.20% |
