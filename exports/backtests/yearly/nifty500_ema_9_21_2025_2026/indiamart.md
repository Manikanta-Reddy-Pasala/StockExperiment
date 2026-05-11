# Indiamart Intermesh Ltd. (INDIAMART)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 2091.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 79 |
| ALERT1 | 52 |
| ALERT2 | 51 |
| ALERT2_SKIP | 33 |
| ALERT3 | 150 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 59 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 60 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 63 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 42
- **Target hits / Stop hits / Partials:** 0 / 60 / 3
- **Avg / median % per leg:** 0.04% / -0.84%
- **Sum % (uncompounded):** 2.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 27 | 12 | 44.4% | 0 | 27 | 0 | 0.17% | 4.5% |
| BUY @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.59% | 1.6% |
| BUY @ 3rd Alert (retest2) | 26 | 11 | 42.3% | 0 | 26 | 0 | 0.11% | 2.9% |
| SELL (all) | 36 | 9 | 25.0% | 0 | 33 | 3 | -0.06% | -2.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 36 | 9 | 25.0% | 0 | 33 | 3 | -0.06% | -2.1% |
| retest1 (combined) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.59% | 1.6% |
| retest2 (combined) | 62 | 20 | 32.3% | 0 | 59 | 3 | 0.01% | 0.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 2309.40 | 2271.27 | 2268.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 14:15:00 | 2353.30 | 2316.36 | 2299.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 09:15:00 | 2324.20 | 2325.07 | 2306.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 2324.20 | 2325.07 | 2306.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 2324.20 | 2325.07 | 2306.55 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 15:15:00 | 2288.00 | 2300.17 | 2300.68 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 09:15:00 | 2328.40 | 2305.82 | 2303.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 11:15:00 | 2335.10 | 2315.59 | 2308.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 15:15:00 | 2322.00 | 2323.97 | 2315.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 15:15:00 | 2322.00 | 2323.97 | 2315.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 2322.00 | 2323.97 | 2315.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 2360.10 | 2323.97 | 2315.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 10:15:00 | 2346.80 | 2325.92 | 2317.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 14:15:00 | 2367.70 | 2382.60 | 2383.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 14:15:00 | 2367.70 | 2382.60 | 2383.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 09:15:00 | 2330.40 | 2370.15 | 2377.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 10:15:00 | 2314.70 | 2313.25 | 2328.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-28 10:30:00 | 2315.30 | 2313.25 | 2328.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 2316.20 | 2313.84 | 2327.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:00:00 | 2316.20 | 2313.84 | 2327.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 2319.50 | 2316.57 | 2325.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 15:00:00 | 2319.50 | 2316.57 | 2325.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 2319.00 | 2317.61 | 2324.51 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 12:15:00 | 2345.70 | 2329.01 | 2326.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 09:15:00 | 2395.30 | 2342.04 | 2334.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 14:15:00 | 2355.40 | 2356.27 | 2345.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 14:15:00 | 2355.40 | 2356.27 | 2345.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 2355.40 | 2356.27 | 2345.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 14:30:00 | 2357.50 | 2356.27 | 2345.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 2395.50 | 2408.49 | 2389.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:30:00 | 2394.70 | 2408.49 | 2389.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 2431.00 | 2411.83 | 2398.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 14:45:00 | 2413.80 | 2411.83 | 2398.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 2411.50 | 2416.43 | 2403.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 14:00:00 | 2430.00 | 2418.28 | 2407.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 13:15:00 | 2450.00 | 2471.19 | 2473.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 13:15:00 | 2450.00 | 2471.19 | 2473.30 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 2492.90 | 2474.65 | 2473.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 15:15:00 | 2498.00 | 2486.85 | 2480.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 13:15:00 | 2479.40 | 2489.57 | 2484.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 13:15:00 | 2479.40 | 2489.57 | 2484.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 2479.40 | 2489.57 | 2484.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 14:00:00 | 2479.40 | 2489.57 | 2484.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 2480.20 | 2487.70 | 2484.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 09:45:00 | 2487.30 | 2485.25 | 2483.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 11:15:00 | 2470.90 | 2482.11 | 2482.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 2470.90 | 2482.11 | 2482.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 2454.00 | 2469.43 | 2475.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 2492.00 | 2460.19 | 2465.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 2492.00 | 2460.19 | 2465.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 2492.00 | 2460.19 | 2465.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:45:00 | 2488.40 | 2460.19 | 2465.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 2484.60 | 2465.07 | 2467.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:45:00 | 2487.40 | 2465.07 | 2467.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 11:15:00 | 2490.60 | 2470.18 | 2469.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 14:15:00 | 2495.80 | 2480.76 | 2474.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 12:15:00 | 2507.40 | 2518.47 | 2502.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 13:00:00 | 2507.40 | 2518.47 | 2502.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 2497.30 | 2514.23 | 2502.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:45:00 | 2495.30 | 2514.23 | 2502.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 2489.80 | 2509.35 | 2501.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 2489.80 | 2509.35 | 2501.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 2605.70 | 2631.19 | 2611.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 11:45:00 | 2612.00 | 2631.19 | 2611.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 2594.60 | 2623.87 | 2609.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:00:00 | 2594.60 | 2623.87 | 2609.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 2605.00 | 2615.19 | 2608.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 2602.80 | 2615.19 | 2608.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 2590.80 | 2610.31 | 2607.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:30:00 | 2588.90 | 2610.31 | 2607.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 10:15:00 | 2565.40 | 2601.33 | 2603.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 12:15:00 | 2558.90 | 2578.76 | 2586.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 14:15:00 | 2581.00 | 2575.41 | 2583.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 14:15:00 | 2581.00 | 2575.41 | 2583.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 2581.00 | 2575.41 | 2583.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 15:00:00 | 2581.00 | 2575.41 | 2583.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 2588.20 | 2577.97 | 2583.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:15:00 | 2555.20 | 2577.97 | 2583.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 2550.90 | 2572.55 | 2580.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 11:45:00 | 2536.70 | 2556.13 | 2567.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:45:00 | 2538.00 | 2547.52 | 2558.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 15:15:00 | 2536.40 | 2545.14 | 2552.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 10:30:00 | 2539.50 | 2542.50 | 2549.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 2545.10 | 2543.02 | 2548.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:00:00 | 2545.10 | 2543.02 | 2548.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 2545.60 | 2543.54 | 2548.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:00:00 | 2545.60 | 2543.54 | 2548.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 2541.60 | 2543.15 | 2547.96 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 2586.20 | 2554.50 | 2552.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 2586.20 | 2554.50 | 2552.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 14:15:00 | 2591.10 | 2571.22 | 2562.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 2568.30 | 2574.78 | 2565.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 10:00:00 | 2568.30 | 2574.78 | 2565.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 2578.90 | 2575.60 | 2566.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 11:30:00 | 2580.80 | 2576.54 | 2567.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 12:00:00 | 2580.30 | 2576.54 | 2567.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 2593.00 | 2582.99 | 2574.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 09:15:00 | 2650.90 | 2673.34 | 2675.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 09:15:00 | 2650.90 | 2673.34 | 2675.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 11:15:00 | 2644.20 | 2663.55 | 2670.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 14:15:00 | 2644.40 | 2628.91 | 2641.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 14:15:00 | 2644.40 | 2628.91 | 2641.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 2644.40 | 2628.91 | 2641.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 15:00:00 | 2644.40 | 2628.91 | 2641.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 2655.00 | 2634.13 | 2642.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 2648.60 | 2634.13 | 2642.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 2613.00 | 2629.90 | 2640.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 10:45:00 | 2603.00 | 2622.72 | 2636.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 2626.90 | 2593.84 | 2592.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 09:15:00 | 2626.90 | 2593.84 | 2592.22 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 2566.20 | 2597.17 | 2598.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 13:15:00 | 2555.30 | 2583.54 | 2592.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 10:15:00 | 2579.50 | 2575.14 | 2584.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 11:00:00 | 2579.50 | 2575.14 | 2584.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 2588.50 | 2556.56 | 2564.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 2588.50 | 2556.56 | 2564.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 2585.00 | 2562.25 | 2566.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 2612.20 | 2562.25 | 2566.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 2620.00 | 2573.80 | 2570.91 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 2544.40 | 2584.51 | 2585.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 10:15:00 | 2508.60 | 2569.33 | 2578.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 09:15:00 | 2520.00 | 2498.65 | 2520.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 2520.00 | 2498.65 | 2520.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 2520.00 | 2498.65 | 2520.34 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 2556.90 | 2520.86 | 2519.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-08 10:15:00 | 2562.90 | 2535.98 | 2527.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-11 09:15:00 | 2536.20 | 2548.39 | 2539.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 09:15:00 | 2536.20 | 2548.39 | 2539.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 2536.20 | 2548.39 | 2539.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:45:00 | 2536.60 | 2548.39 | 2539.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 2544.40 | 2547.59 | 2539.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:15:00 | 2538.10 | 2547.59 | 2539.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 2548.40 | 2547.75 | 2540.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 10:00:00 | 2562.90 | 2546.88 | 2542.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 13:15:00 | 2530.10 | 2553.15 | 2547.70 | SL hit (close<static) qty=1.00 sl=2531.30 alert=retest2 |

### Cycle 18 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 2525.00 | 2542.71 | 2543.60 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 2561.30 | 2546.43 | 2545.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 10:15:00 | 2575.30 | 2552.20 | 2547.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 11:15:00 | 2543.40 | 2550.44 | 2547.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 11:15:00 | 2543.40 | 2550.44 | 2547.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 2543.40 | 2550.44 | 2547.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 2543.40 | 2550.44 | 2547.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 2542.80 | 2548.91 | 2547.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:00:00 | 2542.80 | 2548.91 | 2547.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 2544.50 | 2548.03 | 2546.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 14:30:00 | 2549.20 | 2549.12 | 2547.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 11:00:00 | 2548.80 | 2549.95 | 2548.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 12:30:00 | 2550.10 | 2549.52 | 2548.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 14:00:00 | 2548.80 | 2549.38 | 2548.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 2543.20 | 2548.14 | 2547.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:45:00 | 2536.40 | 2548.14 | 2547.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 2551.00 | 2548.71 | 2548.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:15:00 | 2542.90 | 2548.71 | 2548.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 2534.90 | 2545.95 | 2547.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 09:15:00 | 2534.90 | 2545.95 | 2547.04 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 14:15:00 | 2572.00 | 2550.41 | 2548.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 15:15:00 | 2584.80 | 2557.29 | 2551.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 15:15:00 | 2579.80 | 2585.83 | 2573.57 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 09:15:00 | 2607.10 | 2585.83 | 2573.57 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 2648.50 | 2661.33 | 2649.92 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-25 11:15:00 | 2648.50 | 2661.33 | 2649.92 | SL hit (close<ema400) qty=1.00 sl=2649.92 alert=retest1 |

### Cycle 22 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 2638.70 | 2645.27 | 2645.50 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 14:15:00 | 2656.90 | 2645.68 | 2645.47 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 09:15:00 | 2580.00 | 2633.23 | 2639.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 10:15:00 | 2568.20 | 2620.23 | 2633.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 09:15:00 | 2598.40 | 2589.71 | 2609.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:00:00 | 2598.40 | 2589.71 | 2609.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 2598.90 | 2592.73 | 2607.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 13:15:00 | 2588.00 | 2593.54 | 2606.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 15:15:00 | 2590.00 | 2592.51 | 2603.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 2618.00 | 2597.20 | 2603.83 | SL hit (close>static) qty=1.00 sl=2610.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 2578.20 | 2558.87 | 2557.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 2588.30 | 2574.13 | 2569.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 12:15:00 | 2575.00 | 2577.44 | 2572.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 12:15:00 | 2575.00 | 2577.44 | 2572.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 2575.00 | 2577.44 | 2572.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:30:00 | 2563.90 | 2577.44 | 2572.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 2615.00 | 2584.67 | 2577.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 15:15:00 | 2626.00 | 2604.14 | 2600.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 15:15:00 | 2594.00 | 2600.49 | 2601.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 15:15:00 | 2594.00 | 2600.49 | 2601.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 09:15:00 | 2567.50 | 2593.89 | 2598.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 12:15:00 | 2387.10 | 2380.15 | 2401.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 13:15:00 | 2386.50 | 2380.15 | 2401.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 2375.40 | 2367.94 | 2378.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 12:00:00 | 2344.60 | 2364.10 | 2375.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 15:00:00 | 2327.80 | 2356.45 | 2368.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 2336.50 | 2350.40 | 2355.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:15:00 | 2345.60 | 2350.72 | 2355.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 2350.40 | 2350.65 | 2355.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:45:00 | 2347.40 | 2350.65 | 2355.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 2351.60 | 2350.84 | 2354.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:00:00 | 2351.60 | 2350.84 | 2354.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 2370.40 | 2354.76 | 2356.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:00:00 | 2370.40 | 2354.76 | 2356.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 2380.10 | 2359.82 | 2358.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 2380.10 | 2359.82 | 2358.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 2400.80 | 2368.02 | 2362.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 13:15:00 | 2377.00 | 2382.74 | 2374.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 13:45:00 | 2377.00 | 2382.74 | 2374.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 2373.90 | 2380.97 | 2374.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 15:00:00 | 2373.90 | 2380.97 | 2374.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 2365.80 | 2377.94 | 2373.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:15:00 | 2362.80 | 2377.94 | 2373.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 2363.80 | 2375.11 | 2372.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:45:00 | 2347.60 | 2375.11 | 2372.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 2369.40 | 2373.97 | 2372.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:30:00 | 2364.00 | 2373.97 | 2372.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 2373.20 | 2373.82 | 2372.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:45:00 | 2370.10 | 2373.82 | 2372.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 2377.10 | 2374.47 | 2372.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 13:45:00 | 2385.00 | 2376.46 | 2373.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:15:00 | 2392.70 | 2375.93 | 2374.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 11:45:00 | 2380.00 | 2378.45 | 2375.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:00:00 | 2379.60 | 2378.68 | 2376.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 2375.80 | 2378.10 | 2376.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:00:00 | 2375.80 | 2378.10 | 2376.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-07 14:15:00 | 2359.60 | 2374.40 | 2374.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 14:15:00 | 2359.60 | 2374.40 | 2374.62 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 12:15:00 | 2450.00 | 2387.60 | 2379.55 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 14:15:00 | 2389.60 | 2399.21 | 2399.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 15:15:00 | 2380.00 | 2395.37 | 2397.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 09:15:00 | 2405.50 | 2397.40 | 2398.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 2405.50 | 2397.40 | 2398.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 2405.50 | 2397.40 | 2398.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:45:00 | 2401.40 | 2397.40 | 2398.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 10:15:00 | 2411.50 | 2400.22 | 2399.45 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 2363.70 | 2394.51 | 2397.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 2339.40 | 2383.49 | 2392.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 2350.60 | 2347.63 | 2365.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 11:00:00 | 2350.60 | 2347.63 | 2365.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 2348.00 | 2341.51 | 2354.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:30:00 | 2365.80 | 2341.51 | 2354.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 2349.10 | 2343.03 | 2354.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:00:00 | 2349.10 | 2343.03 | 2354.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 2356.10 | 2345.64 | 2354.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:30:00 | 2363.80 | 2345.64 | 2354.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 2366.60 | 2349.83 | 2355.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 13:00:00 | 2366.60 | 2349.83 | 2355.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 2359.20 | 2351.71 | 2355.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 14:45:00 | 2355.50 | 2352.29 | 2355.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 09:15:00 | 2343.20 | 2353.45 | 2355.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 14:15:00 | 2355.10 | 2345.63 | 2347.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 2375.00 | 2347.17 | 2347.66 | SL hit (close>static) qty=1.00 sl=2368.90 alert=retest2 |

### Cycle 33 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 2379.00 | 2353.54 | 2350.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 2407.60 | 2372.06 | 2361.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 09:15:00 | 2441.50 | 2457.13 | 2433.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-29 10:00:00 | 2441.50 | 2457.13 | 2433.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 2467.00 | 2459.11 | 2436.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:30:00 | 2441.40 | 2459.11 | 2436.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 2456.80 | 2470.31 | 2453.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:45:00 | 2454.10 | 2470.31 | 2453.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 2470.00 | 2471.58 | 2460.47 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 2451.20 | 2460.20 | 2460.70 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 12:15:00 | 2479.00 | 2461.85 | 2461.13 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 10:15:00 | 2435.50 | 2456.90 | 2459.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 10:15:00 | 2431.40 | 2441.69 | 2446.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 14:15:00 | 2447.20 | 2436.78 | 2441.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 14:15:00 | 2447.20 | 2436.78 | 2441.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 2447.20 | 2436.78 | 2441.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:30:00 | 2462.20 | 2436.78 | 2441.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 2449.50 | 2439.33 | 2442.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:15:00 | 2450.40 | 2439.33 | 2442.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 2484.20 | 2448.30 | 2446.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 11:15:00 | 2500.00 | 2465.47 | 2454.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 12:15:00 | 2473.90 | 2488.47 | 2476.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 12:15:00 | 2473.90 | 2488.47 | 2476.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 2473.90 | 2488.47 | 2476.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:45:00 | 2473.10 | 2488.47 | 2476.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 2486.20 | 2488.02 | 2477.50 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 2451.00 | 2471.08 | 2472.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 09:15:00 | 2431.30 | 2457.49 | 2465.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 11:15:00 | 2466.60 | 2458.95 | 2464.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 11:15:00 | 2466.60 | 2458.95 | 2464.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 2466.60 | 2458.95 | 2464.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 12:00:00 | 2466.60 | 2458.95 | 2464.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 2474.30 | 2462.02 | 2465.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:00:00 | 2474.30 | 2462.02 | 2465.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 2475.60 | 2464.74 | 2466.45 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 15:15:00 | 2477.90 | 2469.25 | 2468.32 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 2445.00 | 2464.40 | 2466.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 10:15:00 | 2441.10 | 2459.74 | 2463.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 11:15:00 | 2320.90 | 2318.70 | 2340.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 12:00:00 | 2320.90 | 2318.70 | 2340.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 2336.40 | 2324.53 | 2339.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:00:00 | 2336.40 | 2324.53 | 2339.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 2353.90 | 2330.40 | 2340.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 2353.90 | 2330.40 | 2340.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 2361.90 | 2336.70 | 2342.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 09:15:00 | 2335.20 | 2336.70 | 2342.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 11:15:00 | 2336.60 | 2321.46 | 2320.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 11:15:00 | 2336.60 | 2321.46 | 2320.65 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 10:15:00 | 2313.40 | 2321.74 | 2321.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 13:15:00 | 2302.30 | 2316.09 | 2319.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 14:15:00 | 2310.30 | 2303.32 | 2308.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 14:15:00 | 2310.30 | 2303.32 | 2308.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 2310.30 | 2303.32 | 2308.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:00:00 | 2310.30 | 2303.32 | 2308.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 2319.00 | 2306.45 | 2309.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 2290.00 | 2306.45 | 2309.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 11:15:00 | 2272.10 | 2245.57 | 2243.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 11:15:00 | 2272.10 | 2245.57 | 2243.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 13:15:00 | 2280.20 | 2256.24 | 2249.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 2247.20 | 2256.10 | 2250.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 2247.20 | 2256.10 | 2250.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 2247.20 | 2256.10 | 2250.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:45:00 | 2246.40 | 2256.10 | 2250.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 2275.30 | 2259.94 | 2253.18 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 13:15:00 | 2234.50 | 2249.09 | 2249.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 14:15:00 | 2222.90 | 2243.85 | 2247.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 14:15:00 | 2240.00 | 2232.63 | 2238.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 14:15:00 | 2240.00 | 2232.63 | 2238.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 2240.00 | 2232.63 | 2238.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:00:00 | 2240.00 | 2232.63 | 2238.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 2238.70 | 2233.85 | 2238.28 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 12:15:00 | 2259.10 | 2241.65 | 2240.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 13:15:00 | 2271.30 | 2247.58 | 2243.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 10:15:00 | 2245.00 | 2252.89 | 2247.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 10:15:00 | 2245.00 | 2252.89 | 2247.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 2245.00 | 2252.89 | 2247.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:00:00 | 2245.00 | 2252.89 | 2247.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 2238.50 | 2250.01 | 2247.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:00:00 | 2238.50 | 2250.01 | 2247.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 12:15:00 | 2224.00 | 2244.81 | 2244.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-19 14:15:00 | 2216.60 | 2235.26 | 2240.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 15:15:00 | 2218.00 | 2217.40 | 2226.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:15:00 | 2219.90 | 2217.40 | 2226.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 2242.00 | 2222.32 | 2227.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 10:00:00 | 2242.00 | 2222.32 | 2227.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 2233.30 | 2224.52 | 2227.98 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 13:15:00 | 2259.00 | 2235.54 | 2232.52 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 15:15:00 | 2217.30 | 2235.17 | 2236.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 2206.00 | 2222.57 | 2229.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 13:15:00 | 2214.20 | 2212.62 | 2221.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 13:15:00 | 2214.20 | 2212.62 | 2221.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 2214.20 | 2212.62 | 2221.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 2214.20 | 2212.62 | 2221.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 2197.90 | 2209.68 | 2219.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:30:00 | 2220.00 | 2209.68 | 2219.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 2219.50 | 2208.63 | 2216.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:45:00 | 2222.40 | 2208.63 | 2216.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 2224.00 | 2211.70 | 2216.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:00:00 | 2224.00 | 2211.70 | 2216.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 2246.00 | 2218.56 | 2219.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:45:00 | 2235.10 | 2218.56 | 2219.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 2249.10 | 2224.67 | 2222.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 14:15:00 | 2256.80 | 2231.10 | 2225.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 15:15:00 | 2212.20 | 2227.32 | 2224.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 15:15:00 | 2212.20 | 2227.32 | 2224.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 2212.20 | 2227.32 | 2224.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 2203.50 | 2227.32 | 2224.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 2219.90 | 2225.83 | 2223.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 2212.10 | 2225.83 | 2223.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 2212.00 | 2223.07 | 2222.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 2212.00 | 2223.07 | 2222.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 11:15:00 | 2210.30 | 2220.51 | 2221.56 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 15:15:00 | 2234.00 | 2221.79 | 2221.52 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 09:15:00 | 2212.40 | 2219.91 | 2220.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 12:15:00 | 2198.00 | 2213.66 | 2217.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 09:15:00 | 2182.50 | 2180.61 | 2192.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-05 09:45:00 | 2182.10 | 2180.61 | 2192.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 2195.60 | 2183.60 | 2192.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:30:00 | 2196.10 | 2183.60 | 2192.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 2190.60 | 2185.00 | 2192.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:00:00 | 2190.60 | 2185.00 | 2192.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 2191.80 | 2186.36 | 2192.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 13:00:00 | 2191.80 | 2186.36 | 2192.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 2193.70 | 2187.83 | 2192.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 13:45:00 | 2196.70 | 2187.83 | 2192.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 2201.60 | 2190.58 | 2193.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:45:00 | 2203.70 | 2190.58 | 2193.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 2177.00 | 2187.87 | 2191.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 2174.40 | 2187.87 | 2191.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 10:15:00 | 2174.00 | 2185.99 | 2190.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 10:15:00 | 2159.30 | 2150.07 | 2149.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 10:15:00 | 2159.30 | 2150.07 | 2149.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 14:15:00 | 2162.60 | 2154.50 | 2151.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 15:15:00 | 2150.00 | 2153.60 | 2151.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 15:15:00 | 2150.00 | 2153.60 | 2151.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 2150.00 | 2153.60 | 2151.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 2138.70 | 2153.60 | 2151.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 09:15:00 | 2135.50 | 2149.98 | 2150.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 12:15:00 | 2119.30 | 2138.99 | 2144.75 | Break + close below crossover candle low |

### Cycle 55 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 2282.80 | 2158.90 | 2150.87 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 10:15:00 | 2167.70 | 2196.22 | 2199.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 11:15:00 | 2157.50 | 2188.48 | 2195.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 2223.20 | 2175.51 | 2184.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 2223.20 | 2175.51 | 2184.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 2223.20 | 2175.51 | 2184.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:45:00 | 2233.00 | 2175.51 | 2184.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 2219.30 | 2184.27 | 2187.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 11:00:00 | 2219.30 | 2184.27 | 2187.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2026-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 11:15:00 | 2233.80 | 2194.17 | 2191.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-21 14:15:00 | 2256.70 | 2217.88 | 2204.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-22 11:15:00 | 2219.30 | 2229.84 | 2215.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-22 12:00:00 | 2219.30 | 2229.84 | 2215.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 2195.50 | 2222.97 | 2213.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 13:00:00 | 2195.50 | 2222.97 | 2213.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 2198.90 | 2218.16 | 2212.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 13:45:00 | 2187.10 | 2218.16 | 2212.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 2200.00 | 2213.49 | 2211.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:15:00 | 2218.60 | 2213.49 | 2211.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 2235.40 | 2217.87 | 2213.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-23 10:15:00 | 2244.00 | 2217.87 | 2213.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 13:15:00 | 2196.10 | 2209.79 | 2210.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 2196.10 | 2209.79 | 2210.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 14:15:00 | 2190.20 | 2205.87 | 2209.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 2187.80 | 2185.69 | 2194.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 15:15:00 | 2187.80 | 2185.69 | 2194.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 2187.80 | 2185.69 | 2194.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 2173.30 | 2185.69 | 2194.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 2176.10 | 2183.77 | 2192.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:15:00 | 2168.60 | 2180.72 | 2186.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 11:15:00 | 2170.00 | 2178.90 | 2185.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 12:00:00 | 2168.10 | 2176.74 | 2183.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 12:30:00 | 2169.80 | 2176.31 | 2182.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 2178.50 | 2176.25 | 2181.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:30:00 | 2179.90 | 2176.25 | 2181.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 2175.30 | 2175.84 | 2179.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:30:00 | 2178.30 | 2175.84 | 2179.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 2179.30 | 2176.53 | 2179.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:00:00 | 2179.30 | 2176.53 | 2179.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 2190.00 | 2179.22 | 2180.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:00:00 | 2190.00 | 2179.22 | 2180.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-30 13:15:00 | 2200.90 | 2183.56 | 2182.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 2200.90 | 2183.56 | 2182.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 2230.50 | 2197.75 | 2189.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 2201.20 | 2208.88 | 2197.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 12:15:00 | 2201.20 | 2208.88 | 2197.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 2201.20 | 2208.88 | 2197.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 2216.80 | 2208.88 | 2197.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 2175.00 | 2200.27 | 2197.27 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 2174.70 | 2195.15 | 2195.22 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 2206.50 | 2197.43 | 2196.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 2251.80 | 2208.88 | 2201.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 09:15:00 | 2251.60 | 2251.62 | 2231.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 2251.60 | 2251.62 | 2231.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 2251.60 | 2251.62 | 2231.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 13:15:00 | 2269.10 | 2256.79 | 2239.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 09:15:00 | 2270.20 | 2263.62 | 2247.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 11:45:00 | 2271.00 | 2264.00 | 2251.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 12:15:00 | 2212.00 | 2247.07 | 2250.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 2212.00 | 2247.07 | 2250.91 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 10:15:00 | 2271.90 | 2253.01 | 2250.57 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 2226.40 | 2248.44 | 2249.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 2202.00 | 2234.17 | 2241.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 2209.80 | 2203.30 | 2211.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 09:15:00 | 2209.80 | 2203.30 | 2211.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 2209.80 | 2203.30 | 2211.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:30:00 | 2204.50 | 2203.30 | 2211.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 2203.80 | 2203.40 | 2210.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:30:00 | 2203.30 | 2203.40 | 2210.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 2196.40 | 2202.00 | 2209.26 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 2225.10 | 2211.43 | 2210.61 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 10:15:00 | 2189.70 | 2210.33 | 2211.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 12:15:00 | 2167.00 | 2198.33 | 2205.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 2179.40 | 2168.21 | 2177.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 11:15:00 | 2179.40 | 2168.21 | 2177.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 2179.40 | 2168.21 | 2177.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:00:00 | 2179.40 | 2168.21 | 2177.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 2175.20 | 2169.61 | 2177.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:30:00 | 2184.00 | 2169.61 | 2177.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 2188.70 | 2173.43 | 2178.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:00:00 | 2188.70 | 2173.43 | 2178.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 2187.20 | 2176.18 | 2179.15 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 2200.40 | 2182.60 | 2181.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 11:15:00 | 2210.10 | 2191.84 | 2186.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 10:15:00 | 2202.10 | 2204.47 | 2196.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-24 11:00:00 | 2202.10 | 2204.47 | 2196.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 2202.70 | 2204.11 | 2197.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 11:45:00 | 2197.10 | 2204.11 | 2197.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 12:15:00 | 2211.60 | 2205.61 | 2198.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 12:30:00 | 2200.40 | 2205.61 | 2198.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 2200.80 | 2210.50 | 2204.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 11:45:00 | 2200.10 | 2210.50 | 2204.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 2198.20 | 2208.04 | 2204.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 12:30:00 | 2202.10 | 2208.04 | 2204.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 2200.00 | 2206.43 | 2203.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 14:00:00 | 2200.00 | 2206.43 | 2203.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 2191.00 | 2203.35 | 2202.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 15:00:00 | 2191.00 | 2203.35 | 2202.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 15:15:00 | 2196.00 | 2201.88 | 2201.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 09:15:00 | 2180.90 | 2197.68 | 2200.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 14:15:00 | 2086.60 | 2085.93 | 2110.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 15:00:00 | 2086.60 | 2085.93 | 2110.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 2087.00 | 2074.33 | 2086.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 2085.60 | 2074.33 | 2086.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 2096.00 | 2078.66 | 2087.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 11:00:00 | 2096.00 | 2078.66 | 2087.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 2099.40 | 2082.81 | 2088.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 11:45:00 | 2100.00 | 2082.81 | 2088.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 13:15:00 | 2122.20 | 2096.74 | 2094.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-09 12:15:00 | 2159.60 | 2119.75 | 2107.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 2167.80 | 2170.28 | 2153.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 11:00:00 | 2167.80 | 2170.28 | 2153.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 2148.00 | 2165.83 | 2152.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:45:00 | 2147.00 | 2165.83 | 2152.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 2136.10 | 2159.88 | 2151.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 2136.10 | 2159.88 | 2151.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 15:15:00 | 2123.00 | 2144.34 | 2145.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 2100.00 | 2135.48 | 2141.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 2111.40 | 2103.81 | 2109.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 2111.40 | 2103.81 | 2109.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 2111.40 | 2103.81 | 2109.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 15:00:00 | 2111.40 | 2103.81 | 2109.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 2101.00 | 2103.25 | 2108.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 10:15:00 | 2096.00 | 2102.84 | 2108.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 15:00:00 | 2094.90 | 2088.60 | 2091.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 2088.00 | 2090.88 | 2092.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 15:15:00 | 1991.20 | 2009.10 | 2034.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 15:15:00 | 1990.15 | 2009.10 | 2034.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 15:15:00 | 1983.60 | 2009.10 | 2034.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 1971.00 | 1969.42 | 1991.87 | SL hit (close>ema200) qty=0.50 sl=1969.42 alert=retest2 |

### Cycle 71 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 2032.00 | 1999.90 | 1996.53 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 1970.10 | 1997.39 | 2000.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 15:15:00 | 1960.00 | 1989.91 | 1996.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 13:15:00 | 1967.70 | 1962.34 | 1978.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-30 14:00:00 | 1967.70 | 1962.34 | 1978.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 1996.90 | 1969.25 | 1979.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 15:00:00 | 1996.90 | 1969.25 | 1979.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 1961.20 | 1967.64 | 1978.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 2027.90 | 1967.64 | 1978.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 2064.30 | 1986.97 | 1986.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 2067.70 | 2036.78 | 2023.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 2082.10 | 2097.31 | 2084.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 15:15:00 | 2082.10 | 2097.31 | 2084.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 2082.10 | 2097.31 | 2084.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:30:00 | 2108.50 | 2097.85 | 2086.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 10:15:00 | 2105.00 | 2097.85 | 2086.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 11:45:00 | 2105.90 | 2100.22 | 2089.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 12:45:00 | 2106.20 | 2101.24 | 2090.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2088.90 | 2098.68 | 2093.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 2099.90 | 2098.68 | 2093.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 10:15:00 | 2157.90 | 2165.14 | 2165.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 2157.90 | 2165.14 | 2165.37 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 11:15:00 | 2175.00 | 2167.11 | 2166.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 12:15:00 | 2184.50 | 2170.59 | 2167.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 13:15:00 | 2166.00 | 2169.67 | 2167.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 13:15:00 | 2166.00 | 2169.67 | 2167.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 13:15:00 | 2166.00 | 2169.67 | 2167.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 14:00:00 | 2166.00 | 2169.67 | 2167.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 2161.10 | 2167.96 | 2167.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 14:30:00 | 2158.30 | 2167.96 | 2167.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 2160.00 | 2166.37 | 2166.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 2140.50 | 2161.19 | 2164.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 12:15:00 | 2138.80 | 2123.10 | 2134.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 12:15:00 | 2138.80 | 2123.10 | 2134.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 2138.80 | 2123.10 | 2134.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:00:00 | 2138.80 | 2123.10 | 2134.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 2119.50 | 2122.38 | 2133.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 09:30:00 | 2106.30 | 2121.78 | 2130.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 10:30:00 | 2113.70 | 2120.91 | 2129.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 13:45:00 | 2114.40 | 2121.28 | 2127.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 15:15:00 | 2113.00 | 2121.54 | 2126.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 2113.00 | 2119.84 | 2125.68 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-29 12:15:00 | 2146.70 | 2126.03 | 2126.74 | SL hit (close>static) qty=1.00 sl=2141.30 alert=retest2 |

### Cycle 77 — BUY (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 13:15:00 | 2135.80 | 2127.98 | 2127.56 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 11:15:00 | 2124.90 | 2127.91 | 2127.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 13:15:00 | 2121.60 | 2126.20 | 2127.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 14:15:00 | 2097.80 | 2087.56 | 2102.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 14:15:00 | 2097.80 | 2087.56 | 2102.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 2097.80 | 2087.56 | 2102.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 15:00:00 | 2097.80 | 2087.56 | 2102.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 15:15:00 | 2098.00 | 2089.64 | 2101.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 2078.30 | 2089.64 | 2101.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 11:00:00 | 2089.20 | 2064.48 | 2068.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 12:00:00 | 2078.80 | 2067.34 | 2069.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 12:15:00 | 2101.60 | 2074.20 | 2072.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2026-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 12:15:00 | 2101.60 | 2074.20 | 2072.36 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-16 09:15:00 | 2360.10 | 2025-05-23 14:15:00 | 2367.70 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2025-05-16 10:15:00 | 2346.80 | 2025-05-23 14:15:00 | 2367.70 | STOP_HIT | 1.00 | 0.89% |
| BUY | retest2 | 2025-06-09 14:00:00 | 2430.00 | 2025-06-13 13:15:00 | 2450.00 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2025-06-18 09:45:00 | 2487.30 | 2025-06-18 11:15:00 | 2470.90 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-07-04 11:45:00 | 2536.70 | 2025-07-09 09:15:00 | 2586.20 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-07-07 09:45:00 | 2538.00 | 2025-07-09 09:15:00 | 2586.20 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-07-07 15:15:00 | 2536.40 | 2025-07-09 09:15:00 | 2586.20 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-07-08 10:30:00 | 2539.50 | 2025-07-09 09:15:00 | 2586.20 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-07-10 11:30:00 | 2580.80 | 2025-07-17 09:15:00 | 2650.90 | STOP_HIT | 1.00 | 2.72% |
| BUY | retest2 | 2025-07-10 12:00:00 | 2580.30 | 2025-07-17 09:15:00 | 2650.90 | STOP_HIT | 1.00 | 2.74% |
| BUY | retest2 | 2025-07-11 09:15:00 | 2593.00 | 2025-07-17 09:15:00 | 2650.90 | STOP_HIT | 1.00 | 2.23% |
| SELL | retest2 | 2025-07-21 10:45:00 | 2603.00 | 2025-07-24 09:15:00 | 2626.90 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-08-12 10:00:00 | 2562.90 | 2025-08-12 13:15:00 | 2530.10 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-08-13 14:30:00 | 2549.20 | 2025-08-18 09:15:00 | 2534.90 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-08-14 11:00:00 | 2548.80 | 2025-08-18 09:15:00 | 2534.90 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-08-14 12:30:00 | 2550.10 | 2025-08-18 09:15:00 | 2534.90 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-08-14 14:00:00 | 2548.80 | 2025-08-18 09:15:00 | 2534.90 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2025-08-20 09:15:00 | 2607.10 | 2025-08-25 11:15:00 | 2648.50 | STOP_HIT | 1.00 | 1.59% |
| SELL | retest2 | 2025-08-29 13:15:00 | 2588.00 | 2025-09-01 09:15:00 | 2618.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-08-29 15:15:00 | 2590.00 | 2025-09-01 09:15:00 | 2618.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-09-02 09:15:00 | 2565.20 | 2025-09-08 09:15:00 | 2578.20 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-09-15 15:15:00 | 2626.00 | 2025-09-16 15:15:00 | 2594.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-09-29 12:00:00 | 2344.60 | 2025-10-01 13:15:00 | 2380.10 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-09-29 15:00:00 | 2327.80 | 2025-10-01 13:15:00 | 2380.10 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-10-01 09:15:00 | 2336.50 | 2025-10-01 13:15:00 | 2380.10 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-10-01 10:15:00 | 2345.60 | 2025-10-01 13:15:00 | 2380.10 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-10-06 13:45:00 | 2385.00 | 2025-10-07 14:15:00 | 2359.60 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-10-07 09:15:00 | 2392.70 | 2025-10-07 14:15:00 | 2359.60 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-10-07 11:45:00 | 2380.00 | 2025-10-07 14:15:00 | 2359.60 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-10-07 13:00:00 | 2379.60 | 2025-10-07 14:15:00 | 2359.60 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-10-16 14:45:00 | 2355.50 | 2025-10-21 13:15:00 | 2375.00 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-10-17 09:15:00 | 2343.20 | 2025-10-21 13:15:00 | 2375.00 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-10-20 14:15:00 | 2355.10 | 2025-10-21 13:15:00 | 2375.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-11-26 09:15:00 | 2335.20 | 2025-12-03 11:15:00 | 2336.60 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-12-08 09:15:00 | 2290.00 | 2025-12-15 11:15:00 | 2272.10 | STOP_HIT | 1.00 | 0.78% |
| SELL | retest2 | 2026-01-06 09:15:00 | 2174.40 | 2026-01-13 10:15:00 | 2159.30 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2026-01-06 10:15:00 | 2174.00 | 2026-01-13 10:15:00 | 2159.30 | STOP_HIT | 1.00 | 0.68% |
| BUY | retest2 | 2026-01-23 10:15:00 | 2244.00 | 2026-01-23 13:15:00 | 2196.10 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2026-01-29 10:15:00 | 2168.60 | 2026-01-30 13:15:00 | 2200.90 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2026-01-29 11:15:00 | 2170.00 | 2026-01-30 13:15:00 | 2200.90 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2026-01-29 12:00:00 | 2168.10 | 2026-01-30 13:15:00 | 2200.90 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2026-01-29 12:30:00 | 2169.80 | 2026-01-30 13:15:00 | 2200.90 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2026-02-04 13:15:00 | 2269.10 | 2026-02-06 12:15:00 | 2212.00 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2026-02-05 09:15:00 | 2270.20 | 2026-02-06 12:15:00 | 2212.00 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2026-02-05 11:45:00 | 2271.00 | 2026-02-06 12:15:00 | 2212.00 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2026-03-17 10:15:00 | 2096.00 | 2026-03-20 15:15:00 | 1991.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 15:00:00 | 2094.90 | 2026-03-20 15:15:00 | 1990.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-19 09:15:00 | 2088.00 | 2026-03-20 15:15:00 | 1983.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-17 10:15:00 | 2096.00 | 2026-03-24 11:15:00 | 1971.00 | STOP_HIT | 0.50 | 5.96% |
| SELL | retest2 | 2026-03-18 15:00:00 | 2094.90 | 2026-03-24 11:15:00 | 1971.00 | STOP_HIT | 0.50 | 5.91% |
| SELL | retest2 | 2026-03-19 09:15:00 | 2088.00 | 2026-03-24 11:15:00 | 1971.00 | STOP_HIT | 0.50 | 5.60% |
| BUY | retest2 | 2026-04-10 09:30:00 | 2108.50 | 2026-04-23 10:15:00 | 2157.90 | STOP_HIT | 1.00 | 2.34% |
| BUY | retest2 | 2026-04-10 10:15:00 | 2105.00 | 2026-04-23 10:15:00 | 2157.90 | STOP_HIT | 1.00 | 2.51% |
| BUY | retest2 | 2026-04-10 11:45:00 | 2105.90 | 2026-04-23 10:15:00 | 2157.90 | STOP_HIT | 1.00 | 2.47% |
| BUY | retest2 | 2026-04-10 12:45:00 | 2106.20 | 2026-04-23 10:15:00 | 2157.90 | STOP_HIT | 1.00 | 2.45% |
| BUY | retest2 | 2026-04-13 10:15:00 | 2099.90 | 2026-04-23 10:15:00 | 2157.90 | STOP_HIT | 1.00 | 2.76% |
| SELL | retest2 | 2026-04-28 09:30:00 | 2106.30 | 2026-04-29 12:15:00 | 2146.70 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2026-04-28 10:30:00 | 2113.70 | 2026-04-29 12:15:00 | 2146.70 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2026-04-28 13:45:00 | 2114.40 | 2026-04-29 12:15:00 | 2146.70 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2026-04-28 15:15:00 | 2113.00 | 2026-04-29 12:15:00 | 2146.70 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-05-05 09:15:00 | 2078.30 | 2026-05-07 12:15:00 | 2101.60 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-05-07 11:00:00 | 2089.20 | 2026-05-07 12:15:00 | 2101.60 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2026-05-07 12:00:00 | 2078.80 | 2026-05-07 12:15:00 | 2101.60 | STOP_HIT | 1.00 | -1.10% |
