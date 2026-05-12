# Endurance Technologies Ltd. (ENDURANCE)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 2530.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 82 |
| ALERT1 | 49 |
| ALERT2 | 47 |
| ALERT2_SKIP | 27 |
| ALERT3 | 120 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 70 |
| PARTIAL | 11 |
| TARGET_HIT | 5 |
| STOP_HIT | 66 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 81 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 29 / 52
- **Target hits / Stop hits / Partials:** 4 / 66 / 11
- **Avg / median % per leg:** 0.22% / -1.01%
- **Sum % (uncompounded):** 17.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 29 | 6 | 20.7% | 4 | 25 | 0 | 0.09% | 2.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 29 | 6 | 20.7% | 4 | 25 | 0 | 0.09% | 2.8% |
| SELL (all) | 52 | 23 | 44.2% | 0 | 41 | 11 | 0.29% | 15.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 52 | 23 | 44.2% | 0 | 41 | 11 | 0.29% | 15.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 81 | 29 | 35.8% | 4 | 66 | 11 | 0.22% | 17.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 15:15:00 | 2420.00 | 2424.63 | 2425.15 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 2436.00 | 2426.90 | 2426.14 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 11:15:00 | 2406.50 | 2424.60 | 2425.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 2399.80 | 2419.64 | 2423.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 2440.90 | 2415.89 | 2419.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 2440.90 | 2415.89 | 2419.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 2440.90 | 2415.89 | 2419.29 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 11:15:00 | 2440.00 | 2424.09 | 2422.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 15:15:00 | 2465.00 | 2437.72 | 2429.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 09:15:00 | 2503.80 | 2508.25 | 2478.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 09:30:00 | 2520.90 | 2508.25 | 2478.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 2482.20 | 2502.18 | 2480.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:00:00 | 2482.20 | 2502.18 | 2480.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 2471.00 | 2495.95 | 2479.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 13:00:00 | 2471.00 | 2495.95 | 2479.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 13:15:00 | 2485.00 | 2493.76 | 2480.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 13:30:00 | 2473.80 | 2493.76 | 2480.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 2476.70 | 2490.35 | 2479.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 2476.70 | 2490.35 | 2479.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 2479.00 | 2488.08 | 2479.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:15:00 | 2512.70 | 2488.08 | 2479.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 11:15:00 | 2485.80 | 2508.12 | 2508.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 2485.80 | 2508.12 | 2508.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 12:15:00 | 2463.90 | 2499.28 | 2504.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 2508.60 | 2435.78 | 2454.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 2508.60 | 2435.78 | 2454.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 2508.60 | 2435.78 | 2454.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:00:00 | 2508.60 | 2435.78 | 2454.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 2453.00 | 2439.23 | 2454.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 2431.60 | 2449.05 | 2454.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:30:00 | 2438.70 | 2446.23 | 2452.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:30:00 | 2441.50 | 2443.37 | 2449.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:15:00 | 2437.20 | 2441.93 | 2446.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 2424.00 | 2438.34 | 2444.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:15:00 | 2421.20 | 2438.34 | 2444.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 14:15:00 | 2423.30 | 2433.05 | 2440.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 15:15:00 | 2419.30 | 2432.24 | 2439.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 12:15:00 | 2423.90 | 2432.26 | 2437.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 2414.70 | 2428.75 | 2434.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 13:15:00 | 2408.50 | 2428.75 | 2434.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 15:15:00 | 2400.10 | 2422.68 | 2430.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 2567.00 | 2447.93 | 2440.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 09:15:00 | 2567.00 | 2447.93 | 2440.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 10:15:00 | 2615.00 | 2481.34 | 2456.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 12:15:00 | 2546.70 | 2550.00 | 2517.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-23 13:00:00 | 2546.70 | 2550.00 | 2517.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 2520.30 | 2541.37 | 2523.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:30:00 | 2514.80 | 2541.37 | 2523.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 2504.10 | 2533.91 | 2521.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:30:00 | 2508.80 | 2533.91 | 2521.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 2506.40 | 2528.41 | 2520.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 12:15:00 | 2528.00 | 2528.41 | 2520.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 15:00:00 | 2513.80 | 2521.71 | 2519.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-25 14:15:00 | 2780.80 | 2649.25 | 2592.32 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 14:15:00 | 2779.00 | 2810.38 | 2811.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 09:15:00 | 2730.70 | 2777.66 | 2790.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 2650.70 | 2641.32 | 2679.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-09 10:00:00 | 2650.70 | 2641.32 | 2679.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 2686.20 | 2650.85 | 2669.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:30:00 | 2685.00 | 2650.85 | 2669.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 15:15:00 | 2681.60 | 2657.00 | 2670.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:15:00 | 2649.60 | 2657.00 | 2670.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 2645.60 | 2632.45 | 2642.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:30:00 | 2635.10 | 2632.45 | 2642.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 2637.50 | 2633.46 | 2641.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:45:00 | 2643.90 | 2633.46 | 2641.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 2640.90 | 2634.95 | 2641.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:00:00 | 2640.90 | 2634.95 | 2641.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 2666.30 | 2641.22 | 2643.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:45:00 | 2666.10 | 2641.22 | 2643.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 13:15:00 | 2679.60 | 2648.89 | 2647.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 15:15:00 | 2708.70 | 2666.94 | 2656.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 2701.90 | 2702.70 | 2685.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 14:15:00 | 2693.30 | 2700.03 | 2690.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 2693.30 | 2700.03 | 2690.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 15:15:00 | 2704.90 | 2700.03 | 2690.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 2704.90 | 2701.00 | 2692.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:15:00 | 2679.10 | 2701.00 | 2692.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 2688.40 | 2698.48 | 2691.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:30:00 | 2690.40 | 2698.48 | 2691.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 2702.20 | 2699.23 | 2692.69 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2025-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 12:15:00 | 2678.00 | 2691.24 | 2692.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 13:15:00 | 2670.40 | 2687.07 | 2690.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 2655.00 | 2631.82 | 2643.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 2655.00 | 2631.82 | 2643.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 2655.00 | 2631.82 | 2643.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 2655.00 | 2631.82 | 2643.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 2629.70 | 2631.40 | 2642.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 11:30:00 | 2615.00 | 2629.86 | 2640.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 12:15:00 | 2619.60 | 2629.86 | 2640.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:30:00 | 2601.70 | 2630.20 | 2634.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 13:15:00 | 2484.25 | 2524.66 | 2543.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 13:15:00 | 2488.62 | 2524.66 | 2543.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 15:15:00 | 2471.61 | 2506.29 | 2531.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-04 09:15:00 | 2510.00 | 2507.03 | 2529.23 | SL hit (close>ema200) qty=0.50 sl=2507.03 alert=retest2 |

### Cycle 10 — BUY (started 2025-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 12:15:00 | 2645.00 | 2544.41 | 2541.48 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 12:15:00 | 2544.60 | 2567.48 | 2569.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 2512.80 | 2547.16 | 2556.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 09:15:00 | 2533.70 | 2532.24 | 2545.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 09:15:00 | 2533.70 | 2532.24 | 2545.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 2533.70 | 2532.24 | 2545.34 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 12:15:00 | 2575.60 | 2539.16 | 2535.97 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 2505.00 | 2531.04 | 2533.16 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 10:15:00 | 2542.60 | 2535.41 | 2534.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 11:15:00 | 2587.80 | 2545.89 | 2539.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 14:15:00 | 2529.70 | 2553.59 | 2545.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 14:15:00 | 2529.70 | 2553.59 | 2545.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 2529.70 | 2553.59 | 2545.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 2529.70 | 2553.59 | 2545.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 2530.00 | 2548.87 | 2544.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 09:15:00 | 2598.00 | 2548.87 | 2544.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-18 10:15:00 | 2857.80 | 2682.24 | 2619.17 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2025-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 13:15:00 | 2819.30 | 2845.31 | 2845.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 15:15:00 | 2810.90 | 2833.63 | 2839.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 10:15:00 | 2830.50 | 2829.88 | 2836.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-26 11:00:00 | 2830.50 | 2829.88 | 2836.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 2869.90 | 2837.26 | 2839.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 13:00:00 | 2869.90 | 2837.26 | 2839.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 13:15:00 | 2853.00 | 2840.41 | 2840.33 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 14:15:00 | 2787.80 | 2829.89 | 2835.56 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2025-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 10:15:00 | 2905.60 | 2845.37 | 2840.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-29 10:15:00 | 2912.90 | 2874.94 | 2859.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-29 13:15:00 | 2879.00 | 2881.84 | 2867.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-29 13:45:00 | 2871.90 | 2881.84 | 2867.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 2874.60 | 2880.40 | 2868.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 14:45:00 | 2864.20 | 2880.40 | 2868.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 2855.90 | 2875.50 | 2866.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 09:15:00 | 2913.30 | 2875.50 | 2866.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 2828.10 | 2906.82 | 2895.54 | SL hit (close<static) qty=1.00 sl=2852.20 alert=retest2 |

### Cycle 19 — SELL (started 2025-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 14:15:00 | 2877.70 | 2887.55 | 2888.47 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2025-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 15:15:00 | 2939.90 | 2898.02 | 2893.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 10:15:00 | 2952.00 | 2915.52 | 2902.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 15:15:00 | 2900.00 | 2917.99 | 2909.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 15:15:00 | 2900.00 | 2917.99 | 2909.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 2900.00 | 2917.99 | 2909.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:15:00 | 2946.60 | 2917.99 | 2909.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 10:45:00 | 2927.10 | 2920.00 | 2911.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 11:15:00 | 2930.30 | 2920.00 | 2911.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 11:15:00 | 2944.20 | 2989.49 | 2995.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2025-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 11:15:00 | 2944.20 | 2989.49 | 2995.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 12:15:00 | 2923.20 | 2976.23 | 2988.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 13:15:00 | 2914.70 | 2867.24 | 2882.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 13:15:00 | 2914.70 | 2867.24 | 2882.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 2914.70 | 2867.24 | 2882.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:00:00 | 2914.70 | 2867.24 | 2882.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 2925.00 | 2878.79 | 2885.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 15:15:00 | 2924.00 | 2878.79 | 2885.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 2918.90 | 2894.05 | 2892.13 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 2875.40 | 2890.87 | 2891.41 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2025-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 14:15:00 | 2909.80 | 2893.41 | 2891.26 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 12:15:00 | 2880.00 | 2890.50 | 2891.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 09:15:00 | 2874.00 | 2886.61 | 2889.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 13:15:00 | 2877.10 | 2873.06 | 2880.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-22 14:00:00 | 2877.10 | 2873.06 | 2880.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 2888.10 | 2871.84 | 2878.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:30:00 | 2915.50 | 2871.84 | 2878.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 2898.30 | 2877.14 | 2879.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 10:45:00 | 2895.00 | 2877.14 | 2879.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 2871.80 | 2876.77 | 2879.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:30:00 | 2876.20 | 2876.77 | 2879.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 2873.10 | 2876.03 | 2878.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:45:00 | 2871.00 | 2876.03 | 2878.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 2856.10 | 2872.05 | 2876.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 2847.30 | 2868.40 | 2874.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:15:00 | 2849.20 | 2865.90 | 2872.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 14:15:00 | 2704.93 | 2760.99 | 2799.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 14:15:00 | 2706.74 | 2760.99 | 2799.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-26 10:15:00 | 2753.00 | 2743.48 | 2780.28 | SL hit (close>ema200) qty=0.50 sl=2743.48 alert=retest2 |

### Cycle 26 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 2793.70 | 2754.90 | 2753.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 2820.50 | 2768.02 | 2759.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 12:15:00 | 2942.10 | 2943.60 | 2913.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 12:45:00 | 2941.20 | 2943.60 | 2913.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 2914.00 | 2937.28 | 2915.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 2914.00 | 2937.28 | 2915.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 2921.80 | 2934.18 | 2916.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 2915.00 | 2934.18 | 2916.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 2931.00 | 2933.55 | 2917.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 2922.00 | 2933.55 | 2917.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 2904.60 | 2927.76 | 2916.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:45:00 | 2901.80 | 2927.76 | 2916.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 2920.80 | 2926.37 | 2916.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:30:00 | 2892.40 | 2926.37 | 2916.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 2931.80 | 2927.45 | 2918.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:30:00 | 2921.40 | 2927.45 | 2918.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 2899.80 | 2922.71 | 2919.04 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 10:15:00 | 2889.00 | 2915.97 | 2916.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 2877.00 | 2893.03 | 2902.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 13:15:00 | 2800.10 | 2799.69 | 2824.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 14:00:00 | 2800.10 | 2799.69 | 2824.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 2825.20 | 2804.79 | 2824.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 2825.20 | 2804.79 | 2824.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 2827.80 | 2809.39 | 2825.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 2862.50 | 2809.39 | 2825.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 2921.00 | 2831.72 | 2833.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:00:00 | 2921.00 | 2831.72 | 2833.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 2909.80 | 2847.33 | 2840.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 13:15:00 | 2938.30 | 2886.71 | 2862.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 2918.50 | 2919.48 | 2892.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 13:00:00 | 2918.50 | 2919.48 | 2892.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 2904.00 | 2916.55 | 2896.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 15:00:00 | 2904.00 | 2916.55 | 2896.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 2887.10 | 2910.66 | 2895.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 2939.70 | 2910.66 | 2895.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 10:30:00 | 2917.70 | 2943.15 | 2940.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 11:15:00 | 2910.00 | 2936.52 | 2937.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 11:15:00 | 2910.00 | 2936.52 | 2937.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 12:15:00 | 2904.00 | 2930.02 | 2934.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 2938.10 | 2920.74 | 2927.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 2938.10 | 2920.74 | 2927.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 2938.10 | 2920.74 | 2927.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 2938.10 | 2920.74 | 2927.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 2936.00 | 2923.79 | 2928.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:30:00 | 2938.10 | 2923.79 | 2928.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 2927.10 | 2924.22 | 2927.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:45:00 | 2929.50 | 2924.22 | 2927.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 2923.30 | 2924.04 | 2927.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:45:00 | 2876.60 | 2916.17 | 2922.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 15:15:00 | 2884.80 | 2867.64 | 2865.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2025-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 15:15:00 | 2884.80 | 2867.64 | 2865.88 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 09:15:00 | 2838.20 | 2861.75 | 2863.36 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 10:15:00 | 2875.40 | 2864.48 | 2864.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 11:15:00 | 2895.00 | 2870.59 | 2867.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 2855.10 | 2882.46 | 2876.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 2855.10 | 2882.46 | 2876.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 2855.10 | 2882.46 | 2876.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 2855.10 | 2882.46 | 2876.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 2870.30 | 2880.02 | 2875.82 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 12:15:00 | 2854.50 | 2871.76 | 2872.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 14:15:00 | 2821.00 | 2858.40 | 2866.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 2850.20 | 2839.11 | 2852.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 12:00:00 | 2850.20 | 2839.11 | 2852.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 2875.00 | 2846.29 | 2854.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 2875.00 | 2846.29 | 2854.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 2875.90 | 2852.21 | 2856.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 15:00:00 | 2862.10 | 2854.19 | 2857.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 10:45:00 | 2862.20 | 2832.70 | 2833.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 11:15:00 | 2876.40 | 2841.44 | 2837.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 11:15:00 | 2876.40 | 2841.44 | 2837.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 13:15:00 | 2901.50 | 2860.95 | 2847.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 2709.80 | 2845.54 | 2845.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 2709.80 | 2845.54 | 2845.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 2709.80 | 2845.54 | 2845.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:00:00 | 2709.80 | 2845.54 | 2845.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 10:15:00 | 2700.70 | 2816.57 | 2832.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 12:15:00 | 2669.00 | 2766.95 | 2805.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 15:15:00 | 2696.90 | 2692.76 | 2729.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-17 09:15:00 | 2708.20 | 2692.76 | 2729.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 2711.90 | 2649.31 | 2671.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:30:00 | 2709.80 | 2649.31 | 2671.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 2715.60 | 2662.57 | 2675.23 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 12:15:00 | 2721.00 | 2684.20 | 2683.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 13:15:00 | 2728.50 | 2693.06 | 2687.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 09:15:00 | 2689.20 | 2702.25 | 2694.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 2689.20 | 2702.25 | 2694.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 2689.20 | 2702.25 | 2694.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 11:00:00 | 2736.00 | 2709.00 | 2697.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 12:30:00 | 2727.20 | 2714.85 | 2702.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 09:30:00 | 2757.80 | 2732.58 | 2716.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 2693.30 | 2709.87 | 2711.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2025-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 10:15:00 | 2693.30 | 2709.87 | 2711.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 11:15:00 | 2677.40 | 2703.38 | 2708.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 15:15:00 | 2700.00 | 2686.19 | 2696.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 15:15:00 | 2700.00 | 2686.19 | 2696.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 2700.00 | 2686.19 | 2696.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 2662.80 | 2686.19 | 2696.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 10:00:00 | 2663.00 | 2681.55 | 2693.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 09:15:00 | 2661.50 | 2681.10 | 2684.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 11:00:00 | 2661.70 | 2673.30 | 2680.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 2678.00 | 2672.52 | 2677.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 15:00:00 | 2678.00 | 2672.52 | 2677.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 2670.00 | 2672.02 | 2676.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:15:00 | 2643.60 | 2672.02 | 2676.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 2649.10 | 2667.43 | 2674.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 11:15:00 | 2636.40 | 2663.53 | 2671.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 12:30:00 | 2638.20 | 2656.04 | 2666.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 15:15:00 | 2635.10 | 2654.70 | 2664.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 09:15:00 | 2728.90 | 2666.41 | 2667.71 | SL hit (close>static) qty=1.00 sl=2709.00 alert=retest2 |

### Cycle 38 — BUY (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 10:15:00 | 2704.80 | 2674.09 | 2671.08 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2025-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 11:15:00 | 2650.00 | 2675.22 | 2675.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 2627.70 | 2661.49 | 2668.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 14:15:00 | 2669.90 | 2657.13 | 2662.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 14:15:00 | 2669.90 | 2657.13 | 2662.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 2669.90 | 2657.13 | 2662.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 15:00:00 | 2669.90 | 2657.13 | 2662.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 2660.00 | 2657.70 | 2662.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:15:00 | 2643.00 | 2657.70 | 2662.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 2644.80 | 2655.12 | 2660.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 2632.90 | 2650.36 | 2655.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 13:15:00 | 2669.40 | 2657.58 | 2657.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 2669.40 | 2657.58 | 2657.00 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 2641.20 | 2654.67 | 2655.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 2625.00 | 2647.25 | 2651.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 12:15:00 | 2628.60 | 2627.58 | 2638.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 13:00:00 | 2628.60 | 2627.58 | 2638.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 2628.50 | 2629.32 | 2637.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 15:15:00 | 2618.30 | 2629.32 | 2637.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 2607.50 | 2624.70 | 2633.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 14:15:00 | 2642.10 | 2594.26 | 2592.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2025-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 14:15:00 | 2642.10 | 2594.26 | 2592.91 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 10:15:00 | 2567.40 | 2597.57 | 2599.83 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2025-12-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 14:15:00 | 2620.70 | 2600.95 | 2600.40 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 2579.00 | 2599.61 | 2600.07 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2025-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 13:15:00 | 2614.00 | 2601.83 | 2600.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 09:15:00 | 2641.70 | 2613.68 | 2606.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 10:15:00 | 2656.10 | 2657.24 | 2636.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 10:30:00 | 2656.70 | 2657.24 | 2636.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 2646.00 | 2656.00 | 2645.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 2648.80 | 2656.00 | 2645.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 2635.00 | 2651.80 | 2644.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:00:00 | 2635.00 | 2651.80 | 2644.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 2616.00 | 2644.64 | 2641.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:00:00 | 2616.00 | 2644.64 | 2641.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2025-12-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 12:15:00 | 2615.00 | 2638.71 | 2639.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 14:15:00 | 2596.90 | 2625.69 | 2633.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 10:15:00 | 2563.90 | 2563.37 | 2581.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 10:45:00 | 2565.50 | 2563.37 | 2581.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 2582.20 | 2565.02 | 2577.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 2582.20 | 2565.02 | 2577.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 2589.50 | 2569.91 | 2578.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 2589.50 | 2569.91 | 2578.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 2582.70 | 2573.05 | 2578.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:30:00 | 2582.00 | 2573.05 | 2578.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 2589.10 | 2576.26 | 2579.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:45:00 | 2589.30 | 2576.26 | 2579.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 2576.70 | 2576.35 | 2578.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 15:00:00 | 2562.70 | 2573.58 | 2577.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 15:15:00 | 2590.00 | 2578.06 | 2576.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2025-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 15:15:00 | 2590.00 | 2578.06 | 2576.48 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 10:15:00 | 2550.00 | 2572.56 | 2574.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 13:15:00 | 2543.00 | 2560.04 | 2567.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 09:15:00 | 2564.70 | 2555.55 | 2563.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 09:15:00 | 2564.70 | 2555.55 | 2563.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 2564.70 | 2555.55 | 2563.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:30:00 | 2569.00 | 2555.55 | 2563.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 2561.00 | 2556.64 | 2562.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 11:15:00 | 2558.40 | 2556.64 | 2562.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 2581.30 | 2553.00 | 2557.23 | SL hit (close>static) qty=1.00 sl=2572.00 alert=retest2 |

### Cycle 50 — BUY (started 2026-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 11:15:00 | 2568.50 | 2560.90 | 2560.38 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 2549.20 | 2559.79 | 2560.04 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 11:15:00 | 2566.90 | 2560.01 | 2559.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 10:15:00 | 2583.00 | 2570.90 | 2565.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 2590.00 | 2591.18 | 2580.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 10:15:00 | 2590.00 | 2591.18 | 2580.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 2590.00 | 2591.18 | 2580.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:45:00 | 2591.40 | 2591.18 | 2580.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 2591.80 | 2591.30 | 2581.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 12:15:00 | 2596.20 | 2591.30 | 2581.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 13:00:00 | 2594.90 | 2592.02 | 2582.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 13:45:00 | 2604.30 | 2594.24 | 2584.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 09:45:00 | 2612.20 | 2598.36 | 2589.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 2570.00 | 2599.33 | 2594.97 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-12 09:15:00 | 2570.00 | 2599.33 | 2594.97 | SL hit (close<static) qty=1.00 sl=2571.60 alert=retest2 |

### Cycle 53 — SELL (started 2026-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 10:15:00 | 2554.50 | 2590.36 | 2591.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 09:15:00 | 2547.90 | 2573.06 | 2581.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 2579.00 | 2565.73 | 2573.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 14:15:00 | 2579.00 | 2565.73 | 2573.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 2579.00 | 2565.73 | 2573.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 15:00:00 | 2579.00 | 2565.73 | 2573.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 2599.00 | 2572.38 | 2575.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 2567.90 | 2572.38 | 2575.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 2546.00 | 2567.10 | 2573.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:15:00 | 2543.00 | 2562.88 | 2569.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 15:15:00 | 2523.00 | 2532.21 | 2535.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-22 14:15:00 | 2415.85 | 2434.58 | 2455.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-22 15:15:00 | 2396.85 | 2425.38 | 2449.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 10:15:00 | 2340.30 | 2334.00 | 2359.33 | SL hit (close>ema200) qty=0.50 sl=2334.00 alert=retest2 |

### Cycle 54 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 2397.00 | 2370.36 | 2369.45 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 2353.10 | 2366.91 | 2367.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 10:15:00 | 2345.30 | 2362.59 | 2365.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 13:15:00 | 2368.60 | 2361.67 | 2364.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 13:15:00 | 2368.60 | 2361.67 | 2364.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 2368.60 | 2361.67 | 2364.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 13:45:00 | 2366.30 | 2361.67 | 2364.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 2368.00 | 2362.94 | 2364.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 15:00:00 | 2368.00 | 2362.94 | 2364.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 2403.80 | 2373.36 | 2369.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 10:15:00 | 2416.70 | 2382.03 | 2373.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 2396.70 | 2406.95 | 2392.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 2396.70 | 2406.95 | 2392.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 2396.70 | 2406.95 | 2392.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 2511.40 | 2405.47 | 2403.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 12:30:00 | 2451.40 | 2472.60 | 2457.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 10:15:00 | 2403.40 | 2442.11 | 2447.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 2403.40 | 2442.11 | 2447.28 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 2487.30 | 2450.17 | 2446.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 12:15:00 | 2499.40 | 2468.48 | 2456.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 09:15:00 | 2572.50 | 2582.28 | 2548.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-12 09:30:00 | 2557.60 | 2582.28 | 2548.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 2566.60 | 2577.13 | 2556.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 13:45:00 | 2527.50 | 2577.13 | 2556.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 2560.00 | 2573.71 | 2557.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 15:00:00 | 2560.00 | 2573.71 | 2557.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 2555.00 | 2569.96 | 2556.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:15:00 | 2501.00 | 2569.96 | 2556.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 2517.60 | 2559.49 | 2553.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 2497.40 | 2559.49 | 2553.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 12:15:00 | 2530.30 | 2548.55 | 2549.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 13:15:00 | 2514.70 | 2541.78 | 2546.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 2525.40 | 2521.45 | 2534.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 09:15:00 | 2525.40 | 2521.45 | 2534.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 2525.40 | 2521.45 | 2534.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:30:00 | 2545.30 | 2521.45 | 2534.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 2515.00 | 2499.94 | 2514.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 2513.00 | 2499.94 | 2514.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 2508.20 | 2501.59 | 2513.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:15:00 | 2498.10 | 2517.03 | 2517.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 10:15:00 | 2504.90 | 2477.48 | 2475.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 2504.90 | 2477.48 | 2475.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 11:15:00 | 2543.20 | 2490.62 | 2481.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 2657.00 | 2665.08 | 2628.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 11:30:00 | 2659.30 | 2665.08 | 2628.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 2639.70 | 2660.01 | 2629.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 15:15:00 | 2669.30 | 2642.52 | 2634.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 2610.00 | 2640.30 | 2635.36 | SL hit (close<static) qty=1.00 sl=2623.50 alert=retest2 |

### Cycle 61 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 2554.70 | 2630.35 | 2634.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 10:15:00 | 2527.60 | 2609.80 | 2624.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 2432.20 | 2428.51 | 2460.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 11:15:00 | 2460.10 | 2438.17 | 2459.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 2460.10 | 2438.17 | 2459.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 12:00:00 | 2460.10 | 2438.17 | 2459.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 2471.90 | 2444.91 | 2460.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 13:00:00 | 2471.90 | 2444.91 | 2460.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 13:15:00 | 2488.40 | 2453.61 | 2463.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:00:00 | 2488.40 | 2453.61 | 2463.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 2503.40 | 2469.68 | 2468.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 11:15:00 | 2532.30 | 2485.01 | 2476.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 2487.60 | 2487.66 | 2479.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 2487.60 | 2487.66 | 2479.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 2487.60 | 2487.66 | 2479.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:30:00 | 2479.00 | 2487.66 | 2479.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 2479.00 | 2485.93 | 2479.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 2422.70 | 2485.93 | 2479.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 2420.00 | 2472.74 | 2474.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 2389.00 | 2452.74 | 2463.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 14:15:00 | 2407.60 | 2401.34 | 2429.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-13 14:45:00 | 2408.00 | 2401.34 | 2429.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 2425.10 | 2393.86 | 2415.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 12:00:00 | 2425.10 | 2393.86 | 2415.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 12:15:00 | 2403.30 | 2395.75 | 2414.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 10:30:00 | 2366.40 | 2389.59 | 2404.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 2363.20 | 2389.59 | 2404.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 11:30:00 | 2367.10 | 2357.08 | 2375.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 13:15:00 | 2365.90 | 2359.67 | 2374.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 2290.00 | 2296.46 | 2315.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 2207.20 | 2296.46 | 2315.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 2248.08 | 2279.55 | 2306.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 2245.04 | 2279.55 | 2306.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 2248.74 | 2279.55 | 2306.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 2247.61 | 2279.55 | 2306.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 2215.60 | 2213.82 | 2244.40 | SL hit (close>ema200) qty=0.50 sl=2213.82 alert=retest2 |

### Cycle 64 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 2341.00 | 2263.04 | 2257.81 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 2232.20 | 2281.45 | 2286.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 2173.70 | 2242.53 | 2256.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 2231.90 | 2216.21 | 2236.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 14:00:00 | 2231.90 | 2216.21 | 2236.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 2258.40 | 2224.64 | 2238.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 15:00:00 | 2258.40 | 2224.64 | 2238.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 2235.00 | 2226.72 | 2238.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 10:00:00 | 2222.00 | 2225.77 | 2236.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 2347.30 | 2231.53 | 2221.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 2347.30 | 2231.53 | 2221.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 13:15:00 | 2380.00 | 2304.97 | 2263.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 2348.60 | 2349.68 | 2311.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:45:00 | 2347.60 | 2349.68 | 2311.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2396.20 | 2422.80 | 2385.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:30:00 | 2420.00 | 2404.57 | 2392.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 11:15:00 | 2418.90 | 2405.91 | 2393.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 09:15:00 | 2422.00 | 2408.05 | 2399.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 14:15:00 | 2375.00 | 2392.22 | 2394.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2026-04-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 14:15:00 | 2375.00 | 2392.22 | 2394.49 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 2417.40 | 2397.70 | 2396.61 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2026-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 14:15:00 | 2381.00 | 2394.90 | 2396.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 09:15:00 | 2356.60 | 2386.61 | 2392.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-20 12:15:00 | 2385.00 | 2380.69 | 2387.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 12:15:00 | 2385.00 | 2380.69 | 2387.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 12:15:00 | 2385.00 | 2380.69 | 2387.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 12:45:00 | 2395.00 | 2380.69 | 2387.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 13:15:00 | 2371.00 | 2378.75 | 2385.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 14:45:00 | 2360.40 | 2375.54 | 2383.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 10:15:00 | 2387.30 | 2376.36 | 2381.88 | SL hit (close>static) qty=1.00 sl=2386.70 alert=retest2 |

### Cycle 70 — BUY (started 2026-04-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 14:15:00 | 2395.00 | 2386.61 | 2385.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 15:15:00 | 2400.00 | 2389.29 | 2386.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 13:15:00 | 2394.40 | 2397.87 | 2392.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 13:15:00 | 2394.40 | 2397.87 | 2392.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 2394.40 | 2397.87 | 2392.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 14:00:00 | 2394.40 | 2397.87 | 2392.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 14:15:00 | 2400.00 | 2398.30 | 2393.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 09:15:00 | 2445.10 | 2399.16 | 2394.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 09:15:00 | 2373.90 | 2394.26 | 2396.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 2373.90 | 2394.26 | 2396.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 2342.00 | 2383.81 | 2391.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 11:15:00 | 2346.00 | 2339.44 | 2358.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 12:00:00 | 2346.00 | 2339.44 | 2358.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 2395.90 | 2350.73 | 2361.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:00:00 | 2395.90 | 2350.73 | 2361.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 2387.80 | 2358.15 | 2364.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:15:00 | 2380.20 | 2358.15 | 2364.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 10:15:00 | 2370.00 | 2363.49 | 2362.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 10:15:00 | 2370.00 | 2363.49 | 2362.91 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 13:15:00 | 2353.60 | 2362.30 | 2362.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 14:15:00 | 2325.70 | 2354.98 | 2359.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 14:15:00 | 2329.90 | 2328.28 | 2340.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 14:15:00 | 2329.90 | 2328.28 | 2340.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 2329.90 | 2328.28 | 2340.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:30:00 | 2342.50 | 2328.28 | 2340.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 2358.80 | 2334.66 | 2341.01 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 2362.20 | 2347.23 | 2346.06 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 12:15:00 | 2328.00 | 2343.38 | 2344.41 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2026-05-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 14:15:00 | 2364.50 | 2348.70 | 2346.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 15:15:00 | 2375.00 | 2353.96 | 2349.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 2331.00 | 2349.37 | 2347.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 09:15:00 | 2331.00 | 2349.37 | 2347.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 2331.00 | 2349.37 | 2347.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:00:00 | 2331.00 | 2349.37 | 2347.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 2334.10 | 2346.31 | 2346.40 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 11:15:00 | 2347.90 | 2346.63 | 2346.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 12:15:00 | 2361.10 | 2349.52 | 2347.86 | Break + close above crossover candle high |

### Cycle 79 — SELL (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 13:15:00 | 2328.00 | 2345.22 | 2346.05 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 15:15:00 | 2363.90 | 2348.04 | 2347.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 2370.20 | 2352.47 | 2349.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 10:15:00 | 2339.50 | 2349.88 | 2348.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 10:15:00 | 2339.50 | 2349.88 | 2348.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 2339.50 | 2349.88 | 2348.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:00:00 | 2339.50 | 2349.88 | 2348.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 2349.10 | 2349.72 | 2348.41 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2026-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 12:15:00 | 2334.90 | 2346.76 | 2347.19 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 2359.50 | 2348.79 | 2348.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 2435.10 | 2367.76 | 2356.87 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 09:15:00 | 2101.70 | 2025-05-21 09:15:00 | 2311.76 | TARGET_HIT | 1.00 | 9.99% |
| BUY | retest2 | 2025-06-09 09:15:00 | 2512.70 | 2025-06-12 11:15:00 | 2485.80 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-06-17 09:15:00 | 2431.60 | 2025-06-20 09:15:00 | 2567.00 | STOP_HIT | 1.00 | -5.57% |
| SELL | retest2 | 2025-06-17 10:30:00 | 2438.70 | 2025-06-20 09:15:00 | 2567.00 | STOP_HIT | 1.00 | -5.26% |
| SELL | retest2 | 2025-06-17 12:30:00 | 2441.50 | 2025-06-20 09:15:00 | 2567.00 | STOP_HIT | 1.00 | -5.14% |
| SELL | retest2 | 2025-06-18 11:15:00 | 2437.20 | 2025-06-20 09:15:00 | 2567.00 | STOP_HIT | 1.00 | -5.33% |
| SELL | retest2 | 2025-06-18 12:15:00 | 2421.20 | 2025-06-20 09:15:00 | 2567.00 | STOP_HIT | 1.00 | -6.02% |
| SELL | retest2 | 2025-06-18 14:15:00 | 2423.30 | 2025-06-20 09:15:00 | 2567.00 | STOP_HIT | 1.00 | -5.93% |
| SELL | retest2 | 2025-06-18 15:15:00 | 2419.30 | 2025-06-20 09:15:00 | 2567.00 | STOP_HIT | 1.00 | -6.11% |
| SELL | retest2 | 2025-06-19 12:15:00 | 2423.90 | 2025-06-20 09:15:00 | 2567.00 | STOP_HIT | 1.00 | -5.90% |
| SELL | retest2 | 2025-06-19 13:15:00 | 2408.50 | 2025-06-20 09:15:00 | 2567.00 | STOP_HIT | 1.00 | -6.58% |
| SELL | retest2 | 2025-06-19 15:15:00 | 2400.10 | 2025-06-20 09:15:00 | 2567.00 | STOP_HIT | 1.00 | -6.95% |
| BUY | retest2 | 2025-06-24 12:15:00 | 2528.00 | 2025-06-25 14:15:00 | 2780.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-24 15:00:00 | 2513.80 | 2025-06-25 14:15:00 | 2765.18 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-23 11:30:00 | 2615.00 | 2025-08-01 13:15:00 | 2484.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 12:15:00 | 2619.60 | 2025-08-01 13:15:00 | 2488.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-25 09:30:00 | 2601.70 | 2025-08-01 15:15:00 | 2471.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 11:30:00 | 2615.00 | 2025-08-04 09:15:00 | 2510.00 | STOP_HIT | 0.50 | 4.02% |
| SELL | retest2 | 2025-07-23 12:15:00 | 2619.60 | 2025-08-04 09:15:00 | 2510.00 | STOP_HIT | 0.50 | 4.18% |
| SELL | retest2 | 2025-07-25 09:30:00 | 2601.70 | 2025-08-04 09:15:00 | 2510.00 | STOP_HIT | 0.50 | 3.52% |
| BUY | retest2 | 2025-08-14 09:15:00 | 2598.00 | 2025-08-18 10:15:00 | 2857.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-01 09:15:00 | 2913.30 | 2025-09-02 09:15:00 | 2828.10 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2025-09-02 09:30:00 | 2883.70 | 2025-09-02 10:15:00 | 2845.80 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-09-02 12:00:00 | 2885.20 | 2025-09-02 14:15:00 | 2877.70 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-09-02 12:45:00 | 2882.00 | 2025-09-02 14:15:00 | 2877.70 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2025-09-04 09:15:00 | 2946.60 | 2025-09-11 11:15:00 | 2944.20 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-09-04 10:45:00 | 2927.10 | 2025-09-11 11:15:00 | 2944.20 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2025-09-04 11:15:00 | 2930.30 | 2025-09-11 11:15:00 | 2944.20 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2025-09-24 09:15:00 | 2847.30 | 2025-09-25 14:15:00 | 2704.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 10:15:00 | 2849.20 | 2025-09-25 14:15:00 | 2706.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 09:15:00 | 2847.30 | 2025-09-26 10:15:00 | 2753.00 | STOP_HIT | 0.50 | 3.31% |
| SELL | retest2 | 2025-09-24 10:15:00 | 2849.20 | 2025-09-26 10:15:00 | 2753.00 | STOP_HIT | 0.50 | 3.38% |
| BUY | retest2 | 2025-10-20 09:15:00 | 2939.70 | 2025-10-24 11:15:00 | 2910.00 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-10-24 10:30:00 | 2917.70 | 2025-10-24 11:15:00 | 2910.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-10-29 09:45:00 | 2876.60 | 2025-11-03 15:15:00 | 2884.80 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-11-07 15:00:00 | 2862.10 | 2025-11-12 11:15:00 | 2876.40 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-11-12 10:45:00 | 2862.20 | 2025-11-12 11:15:00 | 2876.40 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-11-20 11:00:00 | 2736.00 | 2025-11-24 10:15:00 | 2693.30 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-11-20 12:30:00 | 2727.20 | 2025-11-24 10:15:00 | 2693.30 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-11-21 09:30:00 | 2757.80 | 2025-11-24 10:15:00 | 2693.30 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-11-25 09:15:00 | 2662.80 | 2025-12-01 09:15:00 | 2728.90 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-11-25 10:00:00 | 2663.00 | 2025-12-01 09:15:00 | 2728.90 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-11-27 09:15:00 | 2661.50 | 2025-12-01 09:15:00 | 2728.90 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2025-11-27 11:00:00 | 2661.70 | 2025-12-01 09:15:00 | 2728.90 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-11-28 11:15:00 | 2636.40 | 2025-12-01 09:15:00 | 2728.90 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2025-11-28 12:30:00 | 2638.20 | 2025-12-01 09:15:00 | 2728.90 | STOP_HIT | 1.00 | -3.44% |
| SELL | retest2 | 2025-11-28 15:15:00 | 2635.10 | 2025-12-01 09:15:00 | 2728.90 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2025-12-05 09:15:00 | 2632.90 | 2025-12-05 13:15:00 | 2669.40 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-12-09 15:15:00 | 2618.30 | 2025-12-12 14:15:00 | 2642.10 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-12-10 10:45:00 | 2607.50 | 2025-12-12 14:15:00 | 2642.10 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-12-30 15:00:00 | 2562.70 | 2025-12-31 15:15:00 | 2590.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-01-02 11:15:00 | 2558.40 | 2026-01-05 09:15:00 | 2581.30 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2026-01-08 12:15:00 | 2596.20 | 2026-01-12 09:15:00 | 2570.00 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2026-01-08 13:00:00 | 2594.90 | 2026-01-12 09:15:00 | 2570.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2026-01-08 13:45:00 | 2604.30 | 2026-01-12 09:15:00 | 2570.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2026-01-09 09:45:00 | 2612.20 | 2026-01-12 09:15:00 | 2570.00 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-01-14 13:15:00 | 2543.00 | 2026-01-22 14:15:00 | 2415.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 15:15:00 | 2523.00 | 2026-01-22 15:15:00 | 2396.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 13:15:00 | 2543.00 | 2026-01-28 10:15:00 | 2340.30 | STOP_HIT | 0.50 | 7.97% |
| SELL | retest2 | 2026-01-19 15:15:00 | 2523.00 | 2026-01-28 10:15:00 | 2340.30 | STOP_HIT | 0.50 | 7.24% |
| BUY | retest2 | 2026-02-03 09:15:00 | 2511.40 | 2026-02-05 10:15:00 | 2403.40 | STOP_HIT | 1.00 | -4.30% |
| BUY | retest2 | 2026-02-04 12:30:00 | 2451.40 | 2026-02-05 10:15:00 | 2403.40 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2026-02-18 09:15:00 | 2498.10 | 2026-02-23 10:15:00 | 2504.90 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2026-02-27 15:15:00 | 2669.30 | 2026-03-02 09:15:00 | 2610.00 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2026-03-02 15:00:00 | 2666.10 | 2026-03-04 09:15:00 | 2554.70 | STOP_HIT | 1.00 | -4.18% |
| SELL | retest2 | 2026-03-17 10:30:00 | 2366.40 | 2026-03-23 09:15:00 | 2248.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-17 11:15:00 | 2363.20 | 2026-03-23 09:15:00 | 2245.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 11:30:00 | 2367.10 | 2026-03-23 09:15:00 | 2248.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 13:15:00 | 2365.90 | 2026-03-23 09:15:00 | 2247.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-17 10:30:00 | 2366.40 | 2026-03-24 12:15:00 | 2215.60 | STOP_HIT | 0.50 | 6.37% |
| SELL | retest2 | 2026-03-17 11:15:00 | 2363.20 | 2026-03-24 12:15:00 | 2215.60 | STOP_HIT | 0.50 | 6.25% |
| SELL | retest2 | 2026-03-18 11:30:00 | 2367.10 | 2026-03-24 12:15:00 | 2215.60 | STOP_HIT | 0.50 | 6.40% |
| SELL | retest2 | 2026-03-18 13:15:00 | 2365.90 | 2026-03-24 12:15:00 | 2215.60 | STOP_HIT | 0.50 | 6.35% |
| SELL | retest2 | 2026-03-23 09:15:00 | 2207.20 | 2026-03-25 10:15:00 | 2341.00 | STOP_HIT | 1.00 | -6.06% |
| SELL | retest2 | 2026-04-06 10:00:00 | 2222.00 | 2026-04-08 09:15:00 | 2347.30 | STOP_HIT | 1.00 | -5.64% |
| BUY | retest2 | 2026-04-15 09:30:00 | 2420.00 | 2026-04-16 14:15:00 | 2375.00 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2026-04-15 11:15:00 | 2418.90 | 2026-04-16 14:15:00 | 2375.00 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2026-04-16 09:15:00 | 2422.00 | 2026-04-16 14:15:00 | 2375.00 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-04-20 14:45:00 | 2360.40 | 2026-04-21 10:15:00 | 2387.30 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2026-04-23 09:15:00 | 2445.10 | 2026-04-24 09:15:00 | 2373.90 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2026-04-27 14:15:00 | 2380.20 | 2026-04-29 10:15:00 | 2370.00 | STOP_HIT | 1.00 | 0.43% |
