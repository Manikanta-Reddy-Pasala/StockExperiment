# Ceat Ltd. (CEATLTD)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 3326.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 158 |
| ALERT1 | 99 |
| ALERT2 | 98 |
| ALERT2_SKIP | 46 |
| ALERT3 | 290 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 10 |
| ENTRY2 | 114 |
| PARTIAL | 18 |
| TARGET_HIT | 10 |
| STOP_HIT | 114 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 142 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 45 / 97
- **Target hits / Stop hits / Partials:** 10 / 114 / 18
- **Avg / median % per leg:** 0.59% / -0.98%
- **Sum % (uncompounded):** 83.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 68 | 12 | 17.6% | 4 | 62 | 2 | -0.15% | -10.1% |
| BUY @ 2nd Alert (retest1) | 12 | 4 | 33.3% | 0 | 10 | 2 | 0.56% | 6.8% |
| BUY @ 3rd Alert (retest2) | 56 | 8 | 14.3% | 4 | 52 | 0 | -0.30% | -16.9% |
| SELL (all) | 74 | 33 | 44.6% | 6 | 52 | 16 | 1.27% | 93.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 74 | 33 | 44.6% | 6 | 52 | 16 | 1.27% | 93.8% |
| retest1 (combined) | 12 | 4 | 33.3% | 0 | 10 | 2 | 0.56% | 6.8% |
| retest2 (combined) | 130 | 41 | 31.5% | 10 | 104 | 16 | 0.59% | 76.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 12:15:00 | 2328.40 | 2304.86 | 2304.28 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 14:15:00 | 2287.00 | 2303.37 | 2303.83 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 10:15:00 | 2321.10 | 2303.72 | 2301.92 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 15:15:00 | 2289.20 | 2301.17 | 2302.31 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 10:15:00 | 2347.60 | 2311.88 | 2307.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 13:15:00 | 2373.95 | 2333.22 | 2318.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 2381.00 | 2391.92 | 2359.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 09:45:00 | 2378.95 | 2391.92 | 2359.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 2401.55 | 2393.84 | 2363.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:45:00 | 2394.90 | 2393.84 | 2363.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 12:15:00 | 2372.00 | 2389.76 | 2367.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 13:00:00 | 2372.00 | 2389.76 | 2367.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 13:15:00 | 2368.05 | 2385.42 | 2367.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 14:00:00 | 2368.05 | 2385.42 | 2367.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 2371.05 | 2382.55 | 2367.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 14:30:00 | 2367.75 | 2382.55 | 2367.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 2369.00 | 2379.84 | 2367.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:15:00 | 2388.50 | 2379.84 | 2367.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:45:00 | 2377.70 | 2378.52 | 2368.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 11:30:00 | 2377.55 | 2379.27 | 2370.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 14:30:00 | 2376.25 | 2380.24 | 2376.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 15:15:00 | 2380.00 | 2380.19 | 2377.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:15:00 | 2390.00 | 2380.19 | 2377.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 2374.45 | 2379.04 | 2377.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:00:00 | 2374.45 | 2379.04 | 2377.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 2381.20 | 2379.47 | 2377.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 11:15:00 | 2396.20 | 2379.47 | 2377.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 14:00:00 | 2389.10 | 2386.77 | 2381.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 15:15:00 | 2399.00 | 2384.43 | 2381.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 09:15:00 | 2366.05 | 2383.08 | 2381.22 | SL hit (close<static) qty=1.00 sl=2370.35 alert=retest2 |

### Cycle 6 — SELL (started 2024-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 09:15:00 | 2368.60 | 2379.28 | 2380.29 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 11:15:00 | 2385.10 | 2381.42 | 2381.14 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 13:15:00 | 2365.85 | 2378.08 | 2379.66 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 15:15:00 | 2395.00 | 2382.73 | 2381.58 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 2365.00 | 2379.97 | 2381.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 11:15:00 | 2364.80 | 2376.14 | 2379.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 10:15:00 | 2365.20 | 2363.56 | 2370.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 10:30:00 | 2363.15 | 2363.56 | 2370.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 11 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 2416.00 | 2372.49 | 2371.27 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 2335.30 | 2377.83 | 2378.38 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 15:15:00 | 2380.05 | 2369.55 | 2369.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 2408.25 | 2377.29 | 2372.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 15:15:00 | 2520.90 | 2520.92 | 2497.92 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 09:15:00 | 2534.65 | 2520.92 | 2497.92 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 11:30:00 | 2532.80 | 2523.70 | 2504.94 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 13:45:00 | 2531.00 | 2525.19 | 2508.90 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 14:15:00 | 2528.15 | 2525.79 | 2510.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 15:00:00 | 2528.15 | 2525.79 | 2510.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 2511.05 | 2522.24 | 2512.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-13 10:15:00 | 2511.05 | 2522.24 | 2512.69 | SL hit (close<ema400) qty=1.00 sl=2512.69 alert=retest1 |

### Cycle 14 — SELL (started 2024-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 09:15:00 | 2508.00 | 2529.70 | 2530.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 12:15:00 | 2504.15 | 2518.02 | 2524.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 2497.75 | 2493.37 | 2502.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 09:15:00 | 2497.75 | 2493.37 | 2502.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 2497.75 | 2493.37 | 2502.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 12:30:00 | 2474.50 | 2493.28 | 2500.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 13:15:00 | 2521.55 | 2498.93 | 2502.36 | SL hit (close>static) qty=1.00 sl=2519.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 15:15:00 | 2521.95 | 2506.76 | 2505.52 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 09:15:00 | 2478.75 | 2501.15 | 2503.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 10:15:00 | 2461.10 | 2474.22 | 2485.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 14:15:00 | 2487.65 | 2471.12 | 2479.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 14:15:00 | 2487.65 | 2471.12 | 2479.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 2487.65 | 2471.12 | 2479.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 14:30:00 | 2485.55 | 2471.12 | 2479.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 15:15:00 | 2466.00 | 2470.10 | 2478.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 09:15:00 | 2451.95 | 2470.10 | 2478.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 14:15:00 | 2500.05 | 2479.05 | 2478.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 14:15:00 | 2500.05 | 2479.05 | 2478.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 15:15:00 | 2519.95 | 2487.23 | 2482.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 09:15:00 | 2776.05 | 2815.89 | 2750.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 09:45:00 | 2777.40 | 2815.89 | 2750.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 2751.00 | 2795.15 | 2752.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 11:30:00 | 2750.00 | 2795.15 | 2752.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 2738.10 | 2783.74 | 2751.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:45:00 | 2726.85 | 2783.74 | 2751.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 2764.00 | 2779.79 | 2752.20 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2024-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 12:15:00 | 2704.55 | 2738.74 | 2741.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-03 13:15:00 | 2690.00 | 2728.99 | 2737.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-05 09:15:00 | 2682.20 | 2682.04 | 2701.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 09:15:00 | 2682.20 | 2682.04 | 2701.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 2682.20 | 2682.04 | 2701.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 10:15:00 | 2710.00 | 2682.04 | 2701.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 2679.80 | 2681.59 | 2699.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 11:15:00 | 2660.70 | 2681.59 | 2699.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 12:15:00 | 2669.40 | 2680.40 | 2697.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 09:45:00 | 2667.20 | 2681.28 | 2691.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-09 13:15:00 | 2730.15 | 2663.39 | 2668.24 | SL hit (close>static) qty=1.00 sl=2713.20 alert=retest2 |

### Cycle 19 — BUY (started 2024-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 14:15:00 | 2722.10 | 2675.13 | 2673.14 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 14:15:00 | 2639.00 | 2674.56 | 2677.26 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 09:15:00 | 2764.60 | 2666.19 | 2665.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-18 09:15:00 | 2819.10 | 2742.67 | 2721.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 14:15:00 | 2775.05 | 2789.66 | 2756.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-18 15:00:00 | 2775.05 | 2789.66 | 2756.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 2681.65 | 2767.47 | 2751.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 2681.65 | 2767.47 | 2751.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 2693.55 | 2752.69 | 2746.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:15:00 | 2680.00 | 2752.69 | 2746.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2024-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 11:15:00 | 2686.00 | 2739.35 | 2741.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-22 09:15:00 | 2625.25 | 2691.09 | 2715.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 09:15:00 | 2646.75 | 2595.56 | 2623.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 2646.75 | 2595.56 | 2623.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 2646.75 | 2595.56 | 2623.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 2646.75 | 2595.56 | 2623.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 2636.55 | 2603.76 | 2624.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 09:15:00 | 2599.50 | 2625.07 | 2628.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 10:00:00 | 2623.00 | 2624.66 | 2628.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 13:00:00 | 2626.10 | 2613.11 | 2617.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 15:00:00 | 2611.00 | 2615.72 | 2618.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 15:15:00 | 2617.00 | 2615.98 | 2617.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 09:15:00 | 2626.45 | 2615.98 | 2617.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 2617.00 | 2616.18 | 2617.83 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-29 10:15:00 | 2686.00 | 2630.15 | 2624.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2024-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 10:15:00 | 2686.00 | 2630.15 | 2624.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 10:15:00 | 2727.95 | 2673.16 | 2651.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 15:15:00 | 2705.25 | 2714.61 | 2684.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 09:15:00 | 2696.50 | 2714.61 | 2684.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 2723.15 | 2716.32 | 2687.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:30:00 | 2687.70 | 2716.32 | 2687.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 2683.55 | 2701.67 | 2690.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 2683.55 | 2701.67 | 2690.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 2705.00 | 2702.34 | 2692.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:15:00 | 2681.90 | 2702.34 | 2692.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 2670.30 | 2695.93 | 2690.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:30:00 | 2674.25 | 2695.93 | 2690.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 2676.50 | 2692.04 | 2688.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:45:00 | 2667.10 | 2692.04 | 2688.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2024-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 11:15:00 | 2661.90 | 2686.01 | 2686.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 12:15:00 | 2656.05 | 2680.02 | 2683.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 10:15:00 | 2665.00 | 2654.26 | 2667.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 10:15:00 | 2665.00 | 2654.26 | 2667.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 2665.00 | 2654.26 | 2667.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 11:00:00 | 2665.00 | 2654.26 | 2667.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 11:15:00 | 2657.60 | 2654.93 | 2666.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-02 12:15:00 | 2650.00 | 2654.93 | 2666.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-02 13:15:00 | 2653.25 | 2655.90 | 2665.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 14:15:00 | 2675.95 | 2660.72 | 2666.28 | SL hit (close>static) qty=1.00 sl=2670.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 11:15:00 | 2683.30 | 2625.51 | 2617.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 09:15:00 | 2697.10 | 2654.66 | 2644.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 09:15:00 | 2744.50 | 2745.36 | 2705.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 12:15:00 | 2709.90 | 2734.86 | 2710.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 12:15:00 | 2709.90 | 2734.86 | 2710.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 13:00:00 | 2709.90 | 2734.86 | 2710.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 13:15:00 | 2724.00 | 2732.69 | 2711.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 14:30:00 | 2728.10 | 2730.98 | 2712.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 09:15:00 | 2684.90 | 2720.41 | 2711.21 | SL hit (close<static) qty=1.00 sl=2702.20 alert=retest2 |

### Cycle 26 — SELL (started 2024-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 11:15:00 | 2660.35 | 2699.04 | 2702.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 12:15:00 | 2650.15 | 2689.26 | 2697.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 2628.40 | 2621.77 | 2644.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 2628.40 | 2621.77 | 2644.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 2628.40 | 2621.77 | 2644.18 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 2722.00 | 2648.34 | 2646.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 10:15:00 | 2737.80 | 2666.24 | 2654.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 2817.50 | 2850.48 | 2818.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 09:15:00 | 2817.50 | 2850.48 | 2818.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 2817.50 | 2850.48 | 2818.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:00:00 | 2817.50 | 2850.48 | 2818.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 2813.65 | 2843.11 | 2817.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:00:00 | 2813.65 | 2843.11 | 2817.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 2830.70 | 2840.63 | 2818.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 12:30:00 | 2832.00 | 2840.53 | 2820.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-26 10:15:00 | 2808.00 | 2834.03 | 2825.72 | SL hit (close<static) qty=1.00 sl=2811.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 2819.00 | 2851.99 | 2853.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 11:15:00 | 2782.95 | 2838.18 | 2847.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-02 09:15:00 | 2797.35 | 2781.68 | 2800.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 09:15:00 | 2797.35 | 2781.68 | 2800.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 2797.35 | 2781.68 | 2800.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:30:00 | 2818.85 | 2781.68 | 2800.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 2809.70 | 2787.28 | 2801.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 11:00:00 | 2809.70 | 2787.28 | 2801.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 2805.75 | 2790.97 | 2801.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 11:30:00 | 2810.10 | 2790.97 | 2801.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 2812.15 | 2795.21 | 2802.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:45:00 | 2810.00 | 2795.21 | 2802.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 2797.90 | 2796.17 | 2801.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 15:15:00 | 2798.05 | 2796.17 | 2801.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 15:15:00 | 2798.05 | 2796.55 | 2801.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 09:15:00 | 2827.40 | 2796.55 | 2801.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 2824.80 | 2802.20 | 2803.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 09:30:00 | 2828.75 | 2802.20 | 2803.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 10:15:00 | 2847.90 | 2811.34 | 2807.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 13:15:00 | 2872.10 | 2832.93 | 2819.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 13:15:00 | 2914.50 | 2916.74 | 2887.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-05 14:00:00 | 2914.50 | 2916.74 | 2887.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 2894.00 | 2912.20 | 2888.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 15:00:00 | 2894.00 | 2912.20 | 2888.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 15:15:00 | 2887.00 | 2907.16 | 2887.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 09:15:00 | 2874.00 | 2907.16 | 2887.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 2839.75 | 2893.68 | 2883.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 2839.75 | 2893.68 | 2883.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 2851.35 | 2885.21 | 2880.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 11:45:00 | 2877.85 | 2884.17 | 2880.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 14:15:00 | 2858.25 | 2876.72 | 2877.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 14:15:00 | 2858.25 | 2876.72 | 2877.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 10:15:00 | 2827.75 | 2861.02 | 2869.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 14:15:00 | 2844.00 | 2841.51 | 2856.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 14:45:00 | 2847.40 | 2841.51 | 2856.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 2859.00 | 2845.01 | 2856.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:15:00 | 2879.95 | 2845.01 | 2856.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 2859.60 | 2847.93 | 2856.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 13:15:00 | 2838.45 | 2853.47 | 2857.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 11:15:00 | 2891.70 | 2863.66 | 2859.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2024-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 11:15:00 | 2891.70 | 2863.66 | 2859.92 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 10:15:00 | 2853.40 | 2860.03 | 2860.37 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 12:15:00 | 2884.70 | 2864.88 | 2862.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 2903.55 | 2880.39 | 2871.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 2965.90 | 2973.88 | 2946.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-17 10:00:00 | 2965.90 | 2973.88 | 2946.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 2961.85 | 2984.60 | 2963.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 15:00:00 | 2961.85 | 2984.60 | 2963.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 15:15:00 | 2968.00 | 2981.28 | 2964.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 09:15:00 | 2984.00 | 2981.28 | 2964.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 13:15:00 | 2948.60 | 2971.29 | 2965.84 | SL hit (close<static) qty=1.00 sl=2955.20 alert=retest2 |

### Cycle 34 — SELL (started 2024-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 15:15:00 | 2933.00 | 2959.15 | 2960.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 2885.00 | 2944.32 | 2954.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 10:15:00 | 2855.95 | 2850.44 | 2886.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 11:00:00 | 2855.95 | 2850.44 | 2886.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 2901.90 | 2861.60 | 2876.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:45:00 | 2915.55 | 2861.60 | 2876.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 2895.25 | 2868.33 | 2877.90 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2024-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 12:15:00 | 2932.30 | 2886.66 | 2884.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 13:15:00 | 2947.95 | 2898.92 | 2890.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 10:15:00 | 3183.85 | 3201.35 | 3153.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-30 11:00:00 | 3183.85 | 3201.35 | 3153.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 3140.30 | 3180.66 | 3158.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 15:00:00 | 3140.30 | 3180.66 | 3158.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 3143.00 | 3173.13 | 3156.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 09:15:00 | 3169.80 | 3173.13 | 3156.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 10:45:00 | 3163.10 | 3171.97 | 3158.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 09:15:00 | 3131.80 | 3172.17 | 3165.97 | SL hit (close<static) qty=1.00 sl=3140.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 10:15:00 | 3114.20 | 3160.57 | 3161.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 11:15:00 | 3090.00 | 3146.46 | 3154.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 10:15:00 | 3127.40 | 3103.62 | 3124.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 10:15:00 | 3127.40 | 3103.62 | 3124.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 3127.40 | 3103.62 | 3124.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 10:45:00 | 3140.05 | 3103.62 | 3124.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 3184.50 | 3119.79 | 3130.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 12:00:00 | 3184.50 | 3119.79 | 3130.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 3093.20 | 3114.48 | 3127.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 13:45:00 | 3085.55 | 3106.49 | 3122.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 2931.27 | 3036.91 | 3082.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 10:15:00 | 2984.10 | 2981.97 | 3026.53 | SL hit (close>ema200) qty=0.50 sl=2981.97 alert=retest2 |

### Cycle 37 — BUY (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 09:15:00 | 3048.50 | 3029.21 | 3027.39 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 11:15:00 | 3006.00 | 3022.38 | 3024.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 12:15:00 | 2990.00 | 3015.91 | 3021.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-11 09:15:00 | 3007.30 | 3000.75 | 3011.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 09:15:00 | 3007.30 | 3000.75 | 3011.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 3007.30 | 3000.75 | 3011.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:00:00 | 3007.30 | 3000.75 | 3011.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 3004.40 | 3001.48 | 3010.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:00:00 | 3004.40 | 3001.48 | 3010.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 3002.30 | 3001.65 | 3009.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:30:00 | 3003.25 | 3001.65 | 3009.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 12:15:00 | 3009.90 | 3003.30 | 3009.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 12:30:00 | 3022.50 | 3003.30 | 3009.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 13:15:00 | 3000.00 | 3002.64 | 3008.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 11:00:00 | 2987.20 | 3003.25 | 3007.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 09:15:00 | 2972.20 | 2998.37 | 3002.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 12:00:00 | 2998.90 | 2992.48 | 2998.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 13:30:00 | 2986.05 | 2992.26 | 2997.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 2893.25 | 2935.07 | 2957.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 12:15:00 | 2869.00 | 2915.92 | 2944.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 14:00:00 | 2872.00 | 2903.03 | 2933.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 09:15:00 | 2833.75 | 2899.97 | 2926.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-10-18 09:15:00 | 2688.48 | 2896.76 | 2922.49 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 14:15:00 | 2986.40 | 2940.34 | 2934.92 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-10-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 09:15:00 | 2892.00 | 2932.07 | 2932.19 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-10-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 10:15:00 | 3001.00 | 2945.86 | 2938.45 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 10:15:00 | 2905.55 | 2938.49 | 2940.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 12:15:00 | 2877.00 | 2918.43 | 2930.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 12:15:00 | 2897.95 | 2888.32 | 2907.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 13:00:00 | 2897.95 | 2888.32 | 2907.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 13:15:00 | 2757.20 | 2745.35 | 2772.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 13:30:00 | 2781.20 | 2745.35 | 2772.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 14:15:00 | 2782.30 | 2752.74 | 2773.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 15:00:00 | 2782.30 | 2752.74 | 2773.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 15:15:00 | 2814.00 | 2764.99 | 2776.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 09:15:00 | 2792.65 | 2764.99 | 2776.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 10:15:00 | 2780.90 | 2770.72 | 2777.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 11:00:00 | 2780.90 | 2770.72 | 2777.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 11:15:00 | 2776.85 | 2771.94 | 2777.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 11:45:00 | 2783.00 | 2771.94 | 2777.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 12:15:00 | 2776.85 | 2772.93 | 2777.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 13:00:00 | 2776.85 | 2772.93 | 2777.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 13:15:00 | 2779.95 | 2774.33 | 2777.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 14:00:00 | 2779.95 | 2774.33 | 2777.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 2779.95 | 2775.45 | 2777.85 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 2815.75 | 2786.40 | 2782.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 11:15:00 | 2830.95 | 2795.31 | 2786.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 09:15:00 | 2800.00 | 2812.91 | 2800.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 09:15:00 | 2800.00 | 2812.91 | 2800.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 2800.00 | 2812.91 | 2800.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:30:00 | 2801.05 | 2812.91 | 2800.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 2786.00 | 2807.53 | 2799.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:00:00 | 2786.00 | 2807.53 | 2799.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 2788.50 | 2803.72 | 2798.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:30:00 | 2786.70 | 2803.72 | 2798.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 13:15:00 | 2806.50 | 2806.12 | 2800.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 13:45:00 | 2807.85 | 2806.12 | 2800.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 2811.95 | 2807.29 | 2801.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 14:45:00 | 2798.10 | 2807.29 | 2801.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 15:15:00 | 2802.00 | 2806.23 | 2801.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:00:00 | 2835.50 | 2812.08 | 2804.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 09:15:00 | 2760.00 | 2798.93 | 2799.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 2760.00 | 2798.93 | 2799.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 10:15:00 | 2750.00 | 2789.15 | 2795.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 2734.85 | 2733.52 | 2754.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 14:00:00 | 2734.85 | 2733.52 | 2754.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 2753.20 | 2737.30 | 2751.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:00:00 | 2753.20 | 2737.30 | 2751.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 2770.60 | 2743.96 | 2753.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 11:00:00 | 2770.60 | 2743.96 | 2753.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 11:15:00 | 2771.90 | 2749.55 | 2754.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 11:30:00 | 2773.70 | 2749.55 | 2754.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2024-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 13:15:00 | 2776.00 | 2758.93 | 2758.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 14:15:00 | 2783.00 | 2763.75 | 2760.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 15:15:00 | 2865.00 | 2865.16 | 2840.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-11 09:15:00 | 2830.50 | 2865.16 | 2840.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 2848.40 | 2861.81 | 2840.90 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2024-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 10:15:00 | 2822.90 | 2839.05 | 2839.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 11:15:00 | 2788.40 | 2828.92 | 2834.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 11:15:00 | 2729.95 | 2722.30 | 2750.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 12:00:00 | 2729.95 | 2722.30 | 2750.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 2749.25 | 2729.75 | 2749.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:30:00 | 2754.95 | 2729.75 | 2749.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 14:15:00 | 2762.00 | 2736.20 | 2750.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 15:00:00 | 2762.00 | 2736.20 | 2750.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 15:15:00 | 2771.90 | 2743.34 | 2752.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:30:00 | 2739.75 | 2747.88 | 2753.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 10:15:00 | 2756.80 | 2747.88 | 2753.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 11:30:00 | 2757.90 | 2753.94 | 2755.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 13:00:00 | 2760.00 | 2755.15 | 2755.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 13:15:00 | 2800.35 | 2764.19 | 2759.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2024-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 13:15:00 | 2800.35 | 2764.19 | 2759.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 11:15:00 | 2815.95 | 2781.83 | 2770.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-21 09:15:00 | 2806.40 | 2811.20 | 2791.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 09:15:00 | 2806.40 | 2811.20 | 2791.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 2806.40 | 2811.20 | 2791.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:30:00 | 2805.05 | 2811.20 | 2791.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 2803.15 | 2809.59 | 2792.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:45:00 | 2801.90 | 2809.59 | 2792.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 11:15:00 | 2794.45 | 2806.56 | 2792.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 11:30:00 | 2798.20 | 2806.56 | 2792.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 12:15:00 | 2779.55 | 2801.16 | 2791.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 13:00:00 | 2779.55 | 2801.16 | 2791.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 13:15:00 | 2766.75 | 2794.28 | 2789.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 14:00:00 | 2766.75 | 2794.28 | 2789.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2024-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 14:15:00 | 2754.10 | 2786.24 | 2786.25 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 12:15:00 | 2793.00 | 2785.65 | 2785.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 14:15:00 | 2853.65 | 2802.42 | 2793.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 09:15:00 | 2952.60 | 2954.65 | 2926.76 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 11:30:00 | 2982.30 | 2959.79 | 2933.94 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 15:15:00 | 2974.00 | 2969.87 | 2945.64 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 2981.50 | 2970.34 | 2951.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 11:15:00 | 3006.05 | 2970.34 | 2951.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 12:00:00 | 3016.20 | 2979.52 | 2957.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-02 12:15:00 | 3122.70 | 3072.51 | 3025.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 11:15:00 | 3131.42 | 3102.43 | 3063.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-12-03 14:15:00 | 3104.75 | 3107.20 | 3075.57 | SL hit (close<ema200) qty=0.50 sl=3107.20 alert=retest1 |

### Cycle 50 — SELL (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 09:15:00 | 3099.10 | 3109.06 | 3110.02 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 09:15:00 | 3444.00 | 3167.65 | 3134.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 13:15:00 | 3510.00 | 3334.79 | 3233.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 15:15:00 | 3329.00 | 3348.86 | 3258.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-10 09:15:00 | 3369.70 | 3348.86 | 3258.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 11:15:00 | 3267.00 | 3316.19 | 3264.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 12:00:00 | 3267.00 | 3316.19 | 3264.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 12:15:00 | 3259.00 | 3304.75 | 3263.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 13:00:00 | 3259.00 | 3304.75 | 3263.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 13:15:00 | 3256.95 | 3295.19 | 3263.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 13:45:00 | 3250.95 | 3295.19 | 3263.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 14:15:00 | 3243.80 | 3284.91 | 3261.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 14:30:00 | 3251.00 | 3284.91 | 3261.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2024-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 10:15:00 | 3147.05 | 3230.63 | 3240.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 11:15:00 | 3110.00 | 3206.50 | 3228.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 09:15:00 | 3277.30 | 3189.64 | 3207.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 09:15:00 | 3277.30 | 3189.64 | 3207.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 3277.30 | 3189.64 | 3207.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 09:45:00 | 3299.15 | 3189.64 | 3207.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 3301.45 | 3212.00 | 3215.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 11:00:00 | 3301.45 | 3212.00 | 3215.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2024-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 11:15:00 | 3302.35 | 3230.07 | 3223.50 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2024-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 15:15:00 | 3185.00 | 3217.17 | 3220.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-16 10:15:00 | 3149.30 | 3181.33 | 3197.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-17 09:15:00 | 3163.90 | 3159.41 | 3177.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 09:15:00 | 3163.90 | 3159.41 | 3177.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 3163.90 | 3159.41 | 3177.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 11:15:00 | 3136.15 | 3156.62 | 3174.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 09:45:00 | 3120.05 | 3124.16 | 3147.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 15:15:00 | 2979.34 | 3041.35 | 3075.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 15:15:00 | 2964.05 | 3041.35 | 3075.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-23 11:15:00 | 3043.60 | 3037.80 | 3064.71 | SL hit (close>ema200) qty=0.50 sl=3037.80 alert=retest2 |

### Cycle 55 — BUY (started 2024-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 09:15:00 | 3076.00 | 3056.59 | 3055.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 14:15:00 | 3130.00 | 3079.81 | 3068.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 15:15:00 | 3216.00 | 3224.68 | 3190.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-31 09:15:00 | 3218.20 | 3224.68 | 3190.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 3181.15 | 3215.97 | 3189.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 10:00:00 | 3181.15 | 3215.97 | 3189.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 3206.15 | 3214.01 | 3191.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 14:45:00 | 3232.00 | 3209.60 | 3195.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 09:15:00 | 3225.10 | 3212.96 | 3207.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 12:15:00 | 3197.40 | 3203.48 | 3204.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 12:15:00 | 3197.40 | 3203.48 | 3204.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-02 14:15:00 | 3187.55 | 3199.57 | 3202.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 12:15:00 | 3220.70 | 3186.02 | 3192.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 12:15:00 | 3220.70 | 3186.02 | 3192.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 12:15:00 | 3220.70 | 3186.02 | 3192.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 13:00:00 | 3220.70 | 3186.02 | 3192.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 3209.50 | 3190.71 | 3194.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 13:30:00 | 3221.05 | 3190.71 | 3194.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 3192.20 | 3190.66 | 3193.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 3172.95 | 3190.66 | 3193.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 3161.90 | 3184.91 | 3190.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 10:15:00 | 3151.80 | 3184.91 | 3190.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-01-16 09:15:00 | 2836.62 | 3020.42 | 3044.75 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2025-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 11:15:00 | 3030.55 | 3021.13 | 3020.94 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 12:15:00 | 3017.85 | 3020.47 | 3020.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 09:15:00 | 2998.85 | 3015.94 | 3018.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 11:15:00 | 2893.45 | 2885.30 | 2919.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 11:15:00 | 2893.45 | 2885.30 | 2919.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 2893.45 | 2885.30 | 2919.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:30:00 | 2886.45 | 2885.30 | 2919.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 2926.85 | 2893.61 | 2920.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 12:45:00 | 2928.00 | 2893.61 | 2920.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 2930.50 | 2900.99 | 2921.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 14:00:00 | 2930.50 | 2900.99 | 2921.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 2956.05 | 2912.00 | 2924.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 15:00:00 | 2956.05 | 2912.00 | 2924.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 15:15:00 | 2990.00 | 2927.60 | 2930.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:30:00 | 2945.75 | 2927.22 | 2929.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 15:15:00 | 2931.00 | 2929.48 | 2929.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 10:15:00 | 2798.46 | 2862.27 | 2890.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 10:15:00 | 2784.45 | 2862.27 | 2890.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 11:15:00 | 2870.45 | 2863.91 | 2888.26 | SL hit (close>ema200) qty=0.50 sl=2863.91 alert=retest2 |

### Cycle 59 — BUY (started 2025-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 14:15:00 | 2921.85 | 2887.74 | 2884.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 2938.95 | 2903.73 | 2892.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 12:15:00 | 2897.35 | 2908.87 | 2898.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 12:15:00 | 2897.35 | 2908.87 | 2898.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 12:15:00 | 2897.35 | 2908.87 | 2898.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 12:45:00 | 2895.85 | 2908.87 | 2898.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 2892.65 | 2905.62 | 2898.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 13:45:00 | 2883.25 | 2905.62 | 2898.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 15:15:00 | 2900.00 | 2903.64 | 2898.49 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-01-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-31 13:15:00 | 2865.60 | 2891.78 | 2894.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-31 14:15:00 | 2855.45 | 2884.52 | 2891.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 09:15:00 | 2891.10 | 2882.87 | 2889.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 09:15:00 | 2891.10 | 2882.87 | 2889.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 2891.10 | 2882.87 | 2889.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 09:30:00 | 2904.05 | 2882.87 | 2889.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 2926.85 | 2891.67 | 2892.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 10:45:00 | 2941.50 | 2891.67 | 2892.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 11:15:00 | 2925.00 | 2898.33 | 2895.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 13:15:00 | 2932.30 | 2906.04 | 2899.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 15:15:00 | 2975.00 | 2980.13 | 2950.45 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-04 09:15:00 | 3069.50 | 2980.13 | 2950.45 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 2994.90 | 3028.34 | 2999.24 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-05 09:15:00 | 2994.90 | 3028.34 | 2999.24 | SL hit (close<ema400) qty=1.00 sl=2999.24 alert=retest1 |

### Cycle 62 — SELL (started 2025-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 09:15:00 | 2933.50 | 2980.32 | 2986.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 10:15:00 | 2929.55 | 2970.17 | 2980.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 11:15:00 | 2896.35 | 2886.15 | 2909.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-10 12:00:00 | 2896.35 | 2886.15 | 2909.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 12:15:00 | 2620.00 | 2602.81 | 2632.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:00:00 | 2620.00 | 2602.81 | 2632.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 2645.50 | 2611.35 | 2633.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 14:00:00 | 2645.50 | 2611.35 | 2633.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 2645.35 | 2618.15 | 2634.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 14:45:00 | 2663.35 | 2618.15 | 2634.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 10:15:00 | 2618.20 | 2623.56 | 2633.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 11:00:00 | 2618.20 | 2623.56 | 2633.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 12:15:00 | 2622.05 | 2618.09 | 2629.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 13:00:00 | 2622.05 | 2618.09 | 2629.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 13:15:00 | 2630.00 | 2620.47 | 2629.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 13:45:00 | 2634.30 | 2620.47 | 2629.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2025-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 14:15:00 | 2709.75 | 2638.33 | 2636.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 15:15:00 | 2718.05 | 2654.27 | 2643.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 11:15:00 | 2693.00 | 2695.85 | 2679.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 12:00:00 | 2693.00 | 2695.85 | 2679.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 12:15:00 | 2676.40 | 2691.96 | 2679.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 13:00:00 | 2676.40 | 2691.96 | 2679.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 13:15:00 | 2705.05 | 2694.58 | 2681.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 15:00:00 | 2745.30 | 2704.72 | 2687.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 09:15:00 | 2751.20 | 2706.06 | 2689.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 10:15:00 | 2725.30 | 2708.39 | 2692.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 14:45:00 | 2731.20 | 2726.77 | 2708.38 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 2673.75 | 2716.48 | 2706.92 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-24 09:15:00 | 2673.75 | 2716.48 | 2706.92 | SL hit (close<static) qty=1.00 sl=2675.70 alert=retest2 |

### Cycle 64 — SELL (started 2025-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 11:15:00 | 2678.00 | 2700.96 | 2701.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 14:15:00 | 2659.70 | 2683.13 | 2692.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 11:15:00 | 2668.65 | 2665.90 | 2679.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-25 12:00:00 | 2668.65 | 2665.90 | 2679.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 12:15:00 | 2678.25 | 2668.37 | 2679.25 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2025-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 15:15:00 | 2738.75 | 2685.63 | 2684.71 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 2645.60 | 2677.63 | 2681.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 10:15:00 | 2638.20 | 2669.74 | 2677.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 15:15:00 | 2404.00 | 2402.05 | 2452.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-05 09:15:00 | 2493.20 | 2402.05 | 2452.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 2501.95 | 2422.03 | 2456.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:45:00 | 2509.15 | 2422.03 | 2456.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 2508.00 | 2439.22 | 2461.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:00:00 | 2508.00 | 2439.22 | 2461.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2025-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 13:15:00 | 2532.90 | 2483.06 | 2477.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 2582.70 | 2502.99 | 2487.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 14:15:00 | 2569.85 | 2577.23 | 2541.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 15:00:00 | 2569.85 | 2577.23 | 2541.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 2575.00 | 2599.82 | 2586.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:30:00 | 2589.65 | 2599.82 | 2586.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 2561.00 | 2592.05 | 2583.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 2537.50 | 2592.05 | 2583.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 2538.95 | 2581.43 | 2579.72 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2025-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 10:15:00 | 2553.95 | 2575.94 | 2577.37 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 11:15:00 | 2589.10 | 2573.58 | 2571.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 09:15:00 | 2629.40 | 2597.06 | 2585.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-17 14:15:00 | 2636.95 | 2647.33 | 2629.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-17 15:00:00 | 2636.95 | 2647.33 | 2629.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 2635.00 | 2644.87 | 2630.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 09:15:00 | 2664.15 | 2644.87 | 2630.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-25 14:15:00 | 2930.57 | 2864.79 | 2837.42 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 14:15:00 | 2841.10 | 2863.55 | 2865.11 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 11:15:00 | 2879.80 | 2865.86 | 2865.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 11:15:00 | 2895.10 | 2879.01 | 2873.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 14:15:00 | 2875.00 | 2880.66 | 2875.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 14:15:00 | 2875.00 | 2880.66 | 2875.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 14:15:00 | 2875.00 | 2880.66 | 2875.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 15:00:00 | 2875.00 | 2880.66 | 2875.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 15:15:00 | 2875.05 | 2879.54 | 2875.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 09:15:00 | 2913.90 | 2879.54 | 2875.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 10:30:00 | 2892.20 | 2895.26 | 2889.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 12:15:00 | 2861.45 | 2887.62 | 2886.85 | SL hit (close<static) qty=1.00 sl=2875.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-04-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 13:15:00 | 2864.65 | 2883.03 | 2884.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 2839.10 | 2866.35 | 2875.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-04 14:15:00 | 2867.00 | 2866.48 | 2874.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-04 15:00:00 | 2867.00 | 2866.48 | 2874.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 15:15:00 | 2847.85 | 2862.75 | 2871.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 09:15:00 | 2714.60 | 2862.75 | 2871.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 2578.87 | 2822.30 | 2852.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-08 09:15:00 | 2709.35 | 2692.86 | 2754.40 | SL hit (close>ema200) qty=0.50 sl=2692.86 alert=retest2 |

### Cycle 73 — BUY (started 2025-04-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 14:15:00 | 2826.65 | 2735.95 | 2724.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 2869.80 | 2777.09 | 2745.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-21 14:15:00 | 3024.00 | 3030.80 | 2992.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-21 15:00:00 | 3024.00 | 3030.80 | 2992.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 14:15:00 | 3007.00 | 3019.83 | 3005.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 15:00:00 | 3007.00 | 3019.83 | 3005.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 2998.00 | 3015.46 | 3004.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 3024.80 | 3015.46 | 3004.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 11:00:00 | 3012.00 | 3011.62 | 3004.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 14:45:00 | 3010.30 | 3012.43 | 3007.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-30 11:15:00 | 3327.28 | 3165.66 | 3105.06 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 10:15:00 | 3855.50 | 3863.89 | 3864.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 12:15:00 | 3837.10 | 3856.87 | 3861.20 | Break + close below crossover candle low |

### Cycle 75 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 3963.40 | 3865.93 | 3862.57 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 13:15:00 | 3828.90 | 3859.69 | 3861.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 14:15:00 | 3772.40 | 3825.72 | 3839.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 10:15:00 | 3777.50 | 3768.78 | 3793.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-27 10:30:00 | 3763.90 | 3768.78 | 3793.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 3746.00 | 3764.23 | 3789.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 13:00:00 | 3740.50 | 3759.48 | 3784.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 14:30:00 | 3739.80 | 3752.09 | 3776.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 10:15:00 | 3737.40 | 3747.57 | 3770.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 11:15:00 | 3804.00 | 3760.56 | 3772.24 | SL hit (close>static) qty=1.00 sl=3798.60 alert=retest2 |

### Cycle 77 — BUY (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 11:15:00 | 3816.00 | 3769.04 | 3763.54 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 12:15:00 | 3735.00 | 3766.35 | 3767.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 09:15:00 | 3611.00 | 3725.16 | 3747.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 09:15:00 | 3717.80 | 3647.91 | 3667.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 09:15:00 | 3717.80 | 3647.91 | 3667.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 3717.80 | 3647.91 | 3667.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:45:00 | 3739.50 | 3647.91 | 3667.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 3719.00 | 3662.12 | 3672.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 11:15:00 | 3716.40 | 3662.12 | 3672.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 13:15:00 | 3683.00 | 3679.28 | 3679.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 14:15:00 | 3712.00 | 3685.82 | 3682.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 09:15:00 | 3781.00 | 3790.62 | 3761.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 09:15:00 | 3781.00 | 3790.62 | 3761.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 3781.00 | 3790.62 | 3761.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 10:15:00 | 3847.90 | 3790.62 | 3761.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 11:00:00 | 3807.40 | 3832.54 | 3820.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 12:15:00 | 3775.20 | 3811.64 | 3812.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 3775.20 | 3811.64 | 3812.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 3757.00 | 3800.71 | 3807.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 3643.10 | 3620.59 | 3653.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 3643.10 | 3620.59 | 3653.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 3643.10 | 3620.59 | 3653.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:30:00 | 3653.40 | 3620.59 | 3653.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 3592.80 | 3608.20 | 3630.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:00:00 | 3579.30 | 3602.42 | 3626.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 15:15:00 | 3570.00 | 3543.48 | 3541.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 15:15:00 | 3570.00 | 3543.48 | 3541.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 3639.90 | 3562.77 | 3550.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 3610.30 | 3622.32 | 3594.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 09:45:00 | 3614.30 | 3622.32 | 3594.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 3587.50 | 3615.36 | 3593.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:00:00 | 3587.50 | 3615.36 | 3593.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 3586.00 | 3609.49 | 3593.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:00:00 | 3586.00 | 3609.49 | 3593.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 3572.00 | 3601.99 | 3591.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 13:00:00 | 3572.00 | 3601.99 | 3591.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 3640.00 | 3620.62 | 3606.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 15:00:00 | 3684.50 | 3645.93 | 3626.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 09:30:00 | 3711.90 | 3662.57 | 3637.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 09:30:00 | 3722.10 | 3667.42 | 3652.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 09:30:00 | 3720.50 | 3684.74 | 3669.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 3682.20 | 3690.23 | 3677.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:00:00 | 3682.20 | 3690.23 | 3677.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 3680.90 | 3688.36 | 3677.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:30:00 | 3679.20 | 3688.36 | 3677.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 3680.00 | 3686.69 | 3678.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:15:00 | 3656.90 | 3686.69 | 3678.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 3641.50 | 3677.65 | 3674.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 3637.50 | 3677.65 | 3674.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-04 10:15:00 | 3639.90 | 3670.10 | 3671.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 10:15:00 | 3639.90 | 3670.10 | 3671.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 3612.00 | 3658.48 | 3666.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 14:15:00 | 3682.40 | 3631.28 | 3639.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 14:15:00 | 3682.40 | 3631.28 | 3639.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 3682.40 | 3631.28 | 3639.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 15:00:00 | 3682.40 | 3631.28 | 3639.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 3669.80 | 3638.98 | 3641.95 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 09:15:00 | 3729.20 | 3657.03 | 3649.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 12:15:00 | 3821.00 | 3710.98 | 3678.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 10:15:00 | 3833.00 | 3837.52 | 3791.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 10:45:00 | 3833.10 | 3837.52 | 3791.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 3794.70 | 3832.09 | 3801.02 | EMA400 retest candle locked (from upside) |

### Cycle 84 — SELL (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 11:15:00 | 3709.70 | 3774.67 | 3782.76 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 10:15:00 | 3847.70 | 3781.65 | 3778.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 11:15:00 | 3872.70 | 3799.86 | 3786.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 15:15:00 | 3860.00 | 3874.26 | 3850.01 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 09:15:00 | 3934.50 | 3874.26 | 3850.01 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 13:15:00 | 3897.20 | 3891.99 | 3867.53 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 14:00:00 | 3896.70 | 3892.93 | 3870.18 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 3882.40 | 3890.82 | 3871.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 15:00:00 | 3882.40 | 3890.82 | 3871.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 3889.00 | 3890.46 | 3872.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:15:00 | 3834.60 | 3890.46 | 3872.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 3840.40 | 3880.45 | 3869.95 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-17 09:15:00 | 3840.40 | 3880.45 | 3869.95 | SL hit (close<ema400) qty=1.00 sl=3869.95 alert=retest1 |

### Cycle 86 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 3790.50 | 3854.48 | 3862.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 09:15:00 | 3591.50 | 3786.57 | 3824.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 09:15:00 | 3398.20 | 3392.42 | 3459.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-25 10:00:00 | 3398.20 | 3392.42 | 3459.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 3354.50 | 3331.16 | 3352.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:45:00 | 3353.80 | 3331.16 | 3352.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 3353.00 | 3335.53 | 3352.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 3341.90 | 3335.53 | 3352.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 3312.20 | 3330.86 | 3349.08 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2025-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 15:15:00 | 3380.10 | 3351.74 | 3351.23 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 3337.00 | 3348.79 | 3349.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 3309.90 | 3338.66 | 3344.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 13:15:00 | 3250.00 | 3248.74 | 3277.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 13:30:00 | 3247.80 | 3248.74 | 3277.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 3225.90 | 3210.46 | 3229.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 13:45:00 | 3236.20 | 3210.46 | 3229.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 3224.90 | 3213.35 | 3229.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:45:00 | 3225.10 | 3213.35 | 3229.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 3121.00 | 3125.51 | 3151.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 10:15:00 | 3108.20 | 3125.51 | 3151.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 13:15:00 | 3166.30 | 3137.66 | 3138.25 | SL hit (close>static) qty=1.00 sl=3155.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 14:15:00 | 3151.30 | 3140.39 | 3139.43 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 3121.00 | 3138.40 | 3139.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 12:15:00 | 3112.90 | 3133.30 | 3136.91 | Break + close below crossover candle low |

### Cycle 91 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 3247.30 | 3141.57 | 3137.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 10:15:00 | 3315.50 | 3176.35 | 3154.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 12:15:00 | 3210.80 | 3218.66 | 3196.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 12:30:00 | 3210.00 | 3218.66 | 3196.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 3216.30 | 3215.84 | 3198.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 14:30:00 | 3199.50 | 3215.84 | 3198.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 3210.00 | 3213.58 | 3200.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 3204.10 | 3213.58 | 3200.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 3199.00 | 3210.32 | 3203.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:45:00 | 3197.70 | 3210.32 | 3203.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 3199.80 | 3208.21 | 3203.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:45:00 | 3198.20 | 3208.21 | 3203.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 3201.20 | 3208.46 | 3204.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 3201.20 | 3208.46 | 3204.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 3199.70 | 3206.71 | 3204.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:00:00 | 3199.70 | 3206.71 | 3204.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 13:15:00 | 3181.00 | 3201.57 | 3202.22 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 12:15:00 | 3213.60 | 3199.17 | 3197.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 15:15:00 | 3215.90 | 3206.60 | 3201.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 09:15:00 | 3174.50 | 3200.18 | 3199.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 3174.50 | 3200.18 | 3199.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 3174.50 | 3200.18 | 3199.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 3160.00 | 3200.18 | 3199.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 3175.20 | 3195.18 | 3197.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 12:15:00 | 3163.40 | 3185.66 | 3192.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 13:15:00 | 3135.80 | 3118.61 | 3143.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 13:15:00 | 3135.80 | 3118.61 | 3143.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 13:15:00 | 3135.80 | 3118.61 | 3143.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 13:45:00 | 3130.20 | 3118.61 | 3143.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 3114.40 | 3117.77 | 3141.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 09:15:00 | 3088.40 | 3116.65 | 3138.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 3156.60 | 3132.77 | 3135.07 | SL hit (close>static) qty=1.00 sl=3149.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 3177.80 | 3141.77 | 3138.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 12:15:00 | 3198.80 | 3160.29 | 3148.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 3311.10 | 3312.91 | 3259.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 11:00:00 | 3311.10 | 3312.91 | 3259.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 3340.20 | 3347.14 | 3321.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:45:00 | 3369.00 | 3355.46 | 3336.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 12:45:00 | 3374.90 | 3366.52 | 3346.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 14:30:00 | 3372.90 | 3368.66 | 3360.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 3369.90 | 3368.14 | 3360.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 3364.10 | 3367.34 | 3360.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:00:00 | 3364.10 | 3367.34 | 3360.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 3378.90 | 3369.65 | 3362.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:30:00 | 3360.00 | 3369.65 | 3362.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 3362.00 | 3368.12 | 3362.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:45:00 | 3362.60 | 3368.12 | 3362.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 3361.00 | 3366.69 | 3362.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:30:00 | 3362.10 | 3366.69 | 3362.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 3381.90 | 3369.74 | 3364.17 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 3326.30 | 3356.72 | 3360.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 12:15:00 | 3326.30 | 3356.72 | 3360.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 13:15:00 | 3316.00 | 3348.58 | 3356.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 3315.90 | 3301.27 | 3319.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 09:15:00 | 3315.90 | 3301.27 | 3319.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 3315.90 | 3301.27 | 3319.08 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 15:15:00 | 3345.00 | 3324.18 | 3323.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 3407.90 | 3340.93 | 3331.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 13:15:00 | 3429.90 | 3432.33 | 3403.30 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 09:15:00 | 3445.00 | 3432.36 | 3408.38 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 3426.70 | 3430.62 | 3413.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 13:45:00 | 3434.20 | 3430.93 | 3416.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 09:15:00 | 3395.70 | 3425.76 | 3418.02 | SL hit (close<ema400) qty=1.00 sl=3418.02 alert=retest1 |

### Cycle 98 — SELL (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 14:15:00 | 3377.80 | 3411.69 | 3413.53 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 10:15:00 | 3433.00 | 3414.04 | 3413.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 11:15:00 | 3453.40 | 3421.91 | 3417.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 12:15:00 | 3468.50 | 3470.20 | 3450.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-23 13:00:00 | 3468.50 | 3470.20 | 3450.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 3493.70 | 3479.20 | 3461.71 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 09:15:00 | 3426.80 | 3454.93 | 3456.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 3408.00 | 3434.93 | 3445.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 09:15:00 | 3433.80 | 3380.90 | 3398.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 09:15:00 | 3433.80 | 3380.90 | 3398.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 3433.80 | 3380.90 | 3398.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:00:00 | 3433.80 | 3380.90 | 3398.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 3424.00 | 3389.52 | 3400.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:00:00 | 3406.30 | 3397.87 | 3403.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 3465.00 | 3403.77 | 3403.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 09:15:00 | 3465.00 | 3403.77 | 3403.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 3488.50 | 3453.51 | 3431.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 3466.50 | 3492.45 | 3470.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 3466.50 | 3492.45 | 3470.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 3466.50 | 3492.45 | 3470.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:30:00 | 3479.30 | 3492.45 | 3470.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 3457.40 | 3485.44 | 3469.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 3457.00 | 3485.44 | 3469.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 3452.30 | 3478.81 | 3468.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:00:00 | 3452.30 | 3478.81 | 3468.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 3474.10 | 3472.54 | 3467.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 14:30:00 | 3466.70 | 3472.54 | 3467.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 3466.60 | 3471.35 | 3467.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:15:00 | 3483.90 | 3471.35 | 3467.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 3467.20 | 3470.52 | 3467.29 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2025-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 12:15:00 | 3455.00 | 3464.60 | 3465.17 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 14:15:00 | 3470.00 | 3465.44 | 3465.43 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 3451.40 | 3463.97 | 3464.85 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 11:15:00 | 3512.50 | 3473.06 | 3468.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 13:15:00 | 3563.20 | 3547.13 | 3528.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 09:15:00 | 3553.60 | 3556.34 | 3538.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 10:00:00 | 3553.60 | 3556.34 | 3538.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 3536.10 | 3552.29 | 3538.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 3536.10 | 3552.29 | 3538.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 3521.80 | 3546.19 | 3536.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:00:00 | 3521.80 | 3546.19 | 3536.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 3514.10 | 3539.77 | 3534.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:45:00 | 3507.40 | 3539.77 | 3534.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2025-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 14:15:00 | 3476.60 | 3524.09 | 3528.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 15:15:00 | 3465.00 | 3512.27 | 3522.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 11:15:00 | 3502.00 | 3500.61 | 3513.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 11:15:00 | 3502.00 | 3500.61 | 3513.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 3502.00 | 3500.61 | 3513.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:45:00 | 3505.80 | 3500.61 | 3513.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 3526.80 | 3505.84 | 3514.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:45:00 | 3524.00 | 3505.84 | 3514.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 3539.40 | 3512.56 | 3517.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:30:00 | 3536.80 | 3512.56 | 3517.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 15:15:00 | 3560.00 | 3527.20 | 3523.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 3601.90 | 3542.14 | 3530.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 11:15:00 | 4183.00 | 4245.60 | 4145.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 12:00:00 | 4183.00 | 4245.60 | 4145.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 4149.80 | 4210.20 | 4152.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:30:00 | 4157.00 | 4210.20 | 4152.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 4120.00 | 4192.16 | 4149.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 4162.60 | 4192.16 | 4149.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 11:15:00 | 4094.90 | 4160.75 | 4144.97 | SL hit (close<static) qty=1.00 sl=4111.70 alert=retest2 |

### Cycle 108 — SELL (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 13:15:00 | 4056.30 | 4125.10 | 4130.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 09:15:00 | 4036.00 | 4076.08 | 4094.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 09:15:00 | 4040.00 | 4022.25 | 4050.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 4040.00 | 4022.25 | 4050.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 4040.00 | 4022.25 | 4050.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:30:00 | 4031.10 | 4022.25 | 4050.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 4132.90 | 4044.38 | 4057.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:00:00 | 4132.90 | 4044.38 | 4057.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 4107.80 | 4057.06 | 4062.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 12:45:00 | 4081.20 | 4062.45 | 4064.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 13:15:00 | 4083.20 | 4066.60 | 4065.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 13:15:00 | 4083.20 | 4066.60 | 4065.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 14:15:00 | 4100.00 | 4073.28 | 4068.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 13:15:00 | 4066.00 | 4089.08 | 4081.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 13:15:00 | 4066.00 | 4089.08 | 4081.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 13:15:00 | 4066.00 | 4089.08 | 4081.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 14:00:00 | 4066.00 | 4089.08 | 4081.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 4026.30 | 4076.53 | 4076.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 4026.30 | 4076.53 | 4076.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2025-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 15:15:00 | 4029.00 | 4067.02 | 4071.89 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 09:15:00 | 4158.70 | 4066.04 | 4065.20 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 14:15:00 | 3994.20 | 4066.01 | 4068.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 14:15:00 | 3968.00 | 4005.30 | 4031.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 3994.40 | 3971.69 | 4000.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:00:00 | 3994.40 | 3971.69 | 4000.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 4034.00 | 3984.15 | 4003.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:00:00 | 4034.00 | 3984.15 | 4003.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 4034.10 | 3994.14 | 4006.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:45:00 | 4005.40 | 3994.14 | 4006.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 4030.00 | 4001.31 | 4008.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 4020.00 | 4001.31 | 4008.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 4070.00 | 4023.67 | 4018.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 11:15:00 | 4096.90 | 4038.32 | 4025.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 10:15:00 | 4061.50 | 4083.81 | 4059.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-11 11:00:00 | 4061.50 | 4083.81 | 4059.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 4096.90 | 4086.43 | 4062.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 11:00:00 | 4104.90 | 4087.83 | 4073.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 4105.00 | 4080.88 | 4075.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 13:15:00 | 4054.00 | 4074.02 | 4074.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 13:15:00 | 4054.00 | 4074.02 | 4074.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 14:15:00 | 4012.40 | 4061.70 | 4069.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 4035.00 | 4009.94 | 4030.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 4035.00 | 4009.94 | 4030.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 4035.00 | 4009.94 | 4030.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 3990.00 | 4027.85 | 4032.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 14:15:00 | 3790.50 | 3873.32 | 3890.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 13:15:00 | 3869.10 | 3858.15 | 3873.20 | SL hit (close>ema200) qty=0.50 sl=3858.15 alert=retest2 |

### Cycle 115 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 3888.00 | 3876.25 | 3876.07 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 3864.90 | 3878.27 | 3878.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 14:15:00 | 3851.50 | 3870.67 | 3875.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 3913.20 | 3876.13 | 3876.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 3913.20 | 3876.13 | 3876.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 3913.20 | 3876.13 | 3876.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:30:00 | 3913.70 | 3876.13 | 3876.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 10:15:00 | 3889.20 | 3878.74 | 3877.69 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 11:15:00 | 3845.00 | 3871.99 | 3874.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 14:15:00 | 3842.90 | 3864.15 | 3870.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 10:15:00 | 3873.30 | 3860.71 | 3866.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 10:15:00 | 3873.30 | 3860.71 | 3866.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 3873.30 | 3860.71 | 3866.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:30:00 | 3871.00 | 3860.71 | 3866.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 3871.30 | 3862.83 | 3867.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:00:00 | 3871.30 | 3862.83 | 3867.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 3890.00 | 3868.26 | 3869.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 13:00:00 | 3890.00 | 3868.26 | 3869.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2025-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 13:15:00 | 3899.90 | 3874.59 | 3871.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 15:15:00 | 3913.00 | 3887.30 | 3878.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 09:15:00 | 3931.00 | 3943.15 | 3918.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-04 10:00:00 | 3931.00 | 3943.15 | 3918.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 3913.30 | 3937.18 | 3918.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 3913.30 | 3937.18 | 3918.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 3887.50 | 3927.24 | 3915.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:45:00 | 3872.80 | 3927.24 | 3915.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 3875.20 | 3916.83 | 3911.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:45:00 | 3877.80 | 3916.83 | 3911.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 14:15:00 | 3882.20 | 3906.41 | 3907.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 3867.00 | 3893.13 | 3898.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 3829.90 | 3829.69 | 3857.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 3829.90 | 3829.69 | 3857.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 3860.00 | 3832.10 | 3851.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 3860.00 | 3832.10 | 3851.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 3821.10 | 3829.90 | 3848.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:45:00 | 3858.30 | 3836.48 | 3849.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 3841.70 | 3837.53 | 3848.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 11:45:00 | 3820.00 | 3834.92 | 3846.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 13:30:00 | 3822.70 | 3825.84 | 3840.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 15:15:00 | 3834.00 | 3787.12 | 3782.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 15:15:00 | 3834.00 | 3787.12 | 3782.08 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 3758.00 | 3781.32 | 3782.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 10:15:00 | 3744.70 | 3770.83 | 3777.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 13:15:00 | 3739.00 | 3710.77 | 3733.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 13:15:00 | 3739.00 | 3710.77 | 3733.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 3739.00 | 3710.77 | 3733.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:00:00 | 3739.00 | 3710.77 | 3733.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 3742.30 | 3717.08 | 3734.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:30:00 | 3738.00 | 3717.08 | 3734.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 3744.00 | 3722.46 | 3735.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 3770.20 | 3722.46 | 3735.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 10:15:00 | 3847.20 | 3758.85 | 3750.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 11:15:00 | 3914.80 | 3790.04 | 3765.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 10:15:00 | 3902.00 | 3913.06 | 3876.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 10:45:00 | 3901.70 | 3913.06 | 3876.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 3904.30 | 3911.31 | 3879.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:45:00 | 3892.00 | 3911.31 | 3879.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 3891.10 | 3903.60 | 3892.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:00:00 | 3891.10 | 3903.60 | 3892.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 3886.00 | 3900.08 | 3891.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:00:00 | 3886.00 | 3900.08 | 3891.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 3877.10 | 3895.49 | 3890.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 3877.10 | 3895.49 | 3890.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 3875.70 | 3891.53 | 3888.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 3872.80 | 3891.53 | 3888.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 09:15:00 | 3852.80 | 3883.78 | 3885.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 10:15:00 | 3843.20 | 3875.67 | 3881.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 11:15:00 | 3886.40 | 3877.81 | 3882.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 11:15:00 | 3886.40 | 3877.81 | 3882.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 3886.40 | 3877.81 | 3882.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:00:00 | 3886.40 | 3877.81 | 3882.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 3860.00 | 3874.25 | 3880.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:30:00 | 3884.80 | 3874.25 | 3880.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 3866.00 | 3872.60 | 3878.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:00:00 | 3866.00 | 3872.60 | 3878.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 3848.70 | 3863.77 | 3872.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 10:15:00 | 3840.00 | 3863.77 | 3872.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 11:15:00 | 3648.00 | 3770.39 | 3815.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 3741.20 | 3737.28 | 3778.57 | SL hit (close>ema200) qty=0.50 sl=3737.28 alert=retest2 |

### Cycle 125 — BUY (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 09:15:00 | 3820.00 | 3789.50 | 3789.15 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 11:15:00 | 3771.00 | 3788.84 | 3789.10 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 3833.90 | 3790.64 | 3788.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 3871.70 | 3806.85 | 3796.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 11:15:00 | 3852.00 | 3857.26 | 3835.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 11:30:00 | 3858.40 | 3857.26 | 3835.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 3826.20 | 3851.05 | 3834.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:45:00 | 3827.80 | 3851.05 | 3834.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 3811.00 | 3843.04 | 3832.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 3811.00 | 3843.04 | 3832.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 3744.70 | 3812.74 | 3820.48 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 09:15:00 | 3856.30 | 3792.00 | 3791.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-08 10:15:00 | 3924.00 | 3818.40 | 3803.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 15:15:00 | 3836.10 | 3838.71 | 3821.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-09 09:15:00 | 3800.10 | 3838.71 | 3821.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 3836.70 | 3838.31 | 3822.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:30:00 | 3783.20 | 3838.31 | 3822.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 3800.70 | 3830.79 | 3820.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:00:00 | 3800.70 | 3830.79 | 3820.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 3777.10 | 3820.05 | 3816.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 12:00:00 | 3777.10 | 3820.05 | 3816.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — SELL (started 2026-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 12:15:00 | 3751.10 | 3806.26 | 3810.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 13:15:00 | 3734.00 | 3791.81 | 3803.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 11:15:00 | 3747.20 | 3744.78 | 3772.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:00:00 | 3747.20 | 3744.78 | 3772.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 3770.00 | 3752.43 | 3769.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 15:00:00 | 3770.00 | 3752.43 | 3769.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 3766.30 | 3755.21 | 3769.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 3791.80 | 3755.21 | 3769.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 3780.00 | 3760.17 | 3770.17 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2026-01-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 12:15:00 | 3811.00 | 3776.68 | 3775.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 13:15:00 | 3826.70 | 3786.69 | 3780.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 12:15:00 | 3820.00 | 3820.75 | 3804.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-14 13:00:00 | 3820.00 | 3820.75 | 3804.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 3817.20 | 3820.09 | 3809.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:00:00 | 3817.20 | 3820.09 | 3809.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 3796.60 | 3817.26 | 3810.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 12:45:00 | 3795.00 | 3817.26 | 3810.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 3803.00 | 3814.40 | 3810.13 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 3787.60 | 3806.74 | 3807.25 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 09:15:00 | 3851.50 | 3815.69 | 3811.27 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 3773.80 | 3828.91 | 3830.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 3736.70 | 3810.47 | 3821.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 3676.00 | 3614.97 | 3681.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 3676.00 | 3614.97 | 3681.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 3676.00 | 3614.97 | 3681.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 3676.00 | 3614.97 | 3681.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 3745.30 | 3641.04 | 3687.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:00:00 | 3745.30 | 3641.04 | 3687.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 3724.40 | 3657.71 | 3690.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 12:45:00 | 3717.60 | 3671.05 | 3694.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 13:30:00 | 3712.00 | 3680.58 | 3696.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:30:00 | 3707.60 | 3698.45 | 3701.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 11:15:00 | 3718.00 | 3703.97 | 3703.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 11:15:00 | 3718.00 | 3703.97 | 3703.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 09:15:00 | 3740.80 | 3715.80 | 3709.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-27 10:15:00 | 3710.90 | 3714.82 | 3710.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 10:15:00 | 3710.90 | 3714.82 | 3710.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 3710.90 | 3714.82 | 3710.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:45:00 | 3700.30 | 3714.82 | 3710.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 11:15:00 | 3716.90 | 3715.24 | 3710.70 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2026-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 13:15:00 | 3681.00 | 3708.10 | 3708.22 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 3729.00 | 3710.05 | 3708.95 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 09:15:00 | 3678.00 | 3703.64 | 3706.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 11:15:00 | 3664.90 | 3690.99 | 3699.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 14:15:00 | 3698.10 | 3686.82 | 3695.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 14:15:00 | 3698.10 | 3686.82 | 3695.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 14:15:00 | 3698.10 | 3686.82 | 3695.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 15:00:00 | 3698.10 | 3686.82 | 3695.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 3692.00 | 3687.86 | 3694.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 3656.00 | 3687.86 | 3694.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 09:15:00 | 3739.70 | 3669.75 | 3676.26 | SL hit (close>static) qty=1.00 sl=3706.20 alert=retest2 |

### Cycle 139 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 3734.90 | 3682.78 | 3681.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 11:15:00 | 3760.10 | 3698.24 | 3688.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 3742.60 | 3750.93 | 3728.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 12:15:00 | 3742.60 | 3750.93 | 3728.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 3742.60 | 3750.93 | 3728.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 3753.80 | 3750.93 | 3728.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 3711.00 | 3743.10 | 3730.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 3672.50 | 3743.10 | 3730.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 3684.70 | 3731.42 | 3726.14 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 3677.50 | 3720.64 | 3721.72 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 3737.40 | 3723.28 | 3721.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 3787.60 | 3736.14 | 3727.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 3874.60 | 3897.48 | 3857.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 3874.60 | 3897.48 | 3857.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 3865.10 | 3891.01 | 3858.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 3865.10 | 3891.01 | 3858.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 3898.00 | 3884.98 | 3864.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 15:15:00 | 3860.00 | 3884.98 | 3864.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 3860.00 | 3879.98 | 3864.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 3885.80 | 3879.98 | 3864.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 3878.10 | 3879.61 | 3865.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 13:00:00 | 3920.40 | 3894.84 | 3876.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 14:15:00 | 3995.30 | 4018.47 | 4019.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 14:15:00 | 3995.30 | 4018.47 | 4019.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 15:15:00 | 3982.00 | 4011.17 | 4015.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 12:15:00 | 4027.40 | 3997.06 | 4005.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 12:15:00 | 4027.40 | 3997.06 | 4005.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 4027.40 | 3997.06 | 4005.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 13:00:00 | 4027.40 | 3997.06 | 4005.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 13:15:00 | 4021.90 | 4002.03 | 4006.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 13:30:00 | 4020.50 | 4002.03 | 4006.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 3894.50 | 3894.99 | 3923.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:30:00 | 3921.90 | 3894.99 | 3923.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 3886.90 | 3887.12 | 3910.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 12:00:00 | 3862.80 | 3880.83 | 3894.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 12:30:00 | 3849.50 | 3876.26 | 3891.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 13:45:00 | 3860.00 | 3872.23 | 3888.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 3669.66 | 3694.46 | 3721.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 3657.02 | 3694.46 | 3721.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 3667.00 | 3694.46 | 3721.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-02 09:15:00 | 3476.52 | 3562.72 | 3632.31 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 143 — BUY (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 12:15:00 | 3478.10 | 3434.86 | 3429.97 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 3297.00 | 3419.56 | 3426.22 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 3517.10 | 3403.19 | 3390.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 10:15:00 | 3535.90 | 3429.73 | 3403.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 3582.90 | 3667.44 | 3586.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 3582.90 | 3667.44 | 3586.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 3582.90 | 3667.44 | 3586.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 3582.90 | 3667.44 | 3586.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 3542.00 | 3642.35 | 3582.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:00:00 | 3542.00 | 3642.35 | 3582.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 3493.10 | 3612.50 | 3574.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 12:00:00 | 3493.10 | 3612.50 | 3574.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2026-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 14:15:00 | 3474.00 | 3544.00 | 3548.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 15:15:00 | 3459.50 | 3527.10 | 3540.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 11:15:00 | 3509.80 | 3498.98 | 3522.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 11:15:00 | 3509.80 | 3498.98 | 3522.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 3509.80 | 3498.98 | 3522.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 11:45:00 | 3510.00 | 3498.98 | 3522.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 3494.10 | 3438.46 | 3459.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 3502.00 | 3438.46 | 3459.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 3514.00 | 3453.57 | 3464.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:15:00 | 3518.30 | 3453.57 | 3464.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 3553.00 | 3473.45 | 3472.33 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 3400.00 | 3503.23 | 3503.83 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 3554.40 | 3460.94 | 3456.24 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 3464.00 | 3481.27 | 3482.72 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 14:15:00 | 3496.10 | 3484.23 | 3483.94 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 3405.00 | 3468.39 | 3476.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 3397.80 | 3454.27 | 3469.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 3354.00 | 3339.45 | 3393.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 3354.00 | 3339.45 | 3393.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 3354.00 | 3339.45 | 3393.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 3385.20 | 3339.45 | 3393.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 14:15:00 | 3384.00 | 3359.24 | 3383.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 15:00:00 | 3384.00 | 3359.24 | 3383.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 15:15:00 | 3390.00 | 3365.39 | 3383.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 3287.00 | 3365.39 | 3383.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 15:15:00 | 3360.00 | 3353.88 | 3366.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 11:00:00 | 3373.30 | 3353.05 | 3354.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 11:15:00 | 3379.10 | 3358.26 | 3356.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2026-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 11:15:00 | 3379.10 | 3358.26 | 3356.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 3585.00 | 3410.80 | 3382.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 3544.70 | 3604.81 | 3567.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 3544.70 | 3604.81 | 3567.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 3544.70 | 3604.81 | 3567.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 3565.50 | 3604.81 | 3567.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 10:15:00 | 3728.60 | 3779.35 | 3782.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 3728.60 | 3779.35 | 3782.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 12:15:00 | 3721.80 | 3759.05 | 3771.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 11:15:00 | 3560.00 | 3558.44 | 3623.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 12:15:00 | 3574.80 | 3558.44 | 3623.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 3552.00 | 3551.54 | 3584.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:45:00 | 3551.60 | 3551.54 | 3584.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 3788.40 | 3589.17 | 3590.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:30:00 | 3918.70 | 3589.17 | 3590.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 10:15:00 | 3730.50 | 3617.44 | 3602.79 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 3495.00 | 3608.31 | 3609.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 3466.00 | 3579.85 | 3596.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 3317.90 | 3288.70 | 3345.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-06 10:00:00 | 3317.90 | 3288.70 | 3345.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 3343.20 | 3305.72 | 3332.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 14:45:00 | 3354.00 | 3305.72 | 3332.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 3370.20 | 3318.62 | 3336.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 3386.20 | 3318.62 | 3336.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2026-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 12:15:00 | 3384.30 | 3350.61 | 3347.41 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2026-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 10:15:00 | 3319.90 | 3343.98 | 3346.23 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-14 09:30:00 | 2281.20 | 2024-05-14 10:15:00 | 2318.20 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2024-05-22 09:15:00 | 2388.50 | 2024-05-27 09:15:00 | 2366.05 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-05-22 09:45:00 | 2377.70 | 2024-05-27 09:15:00 | 2366.05 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2024-05-22 11:30:00 | 2377.55 | 2024-05-27 09:15:00 | 2366.05 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2024-05-23 14:30:00 | 2376.25 | 2024-05-28 09:15:00 | 2368.60 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2024-05-24 11:15:00 | 2396.20 | 2024-05-28 09:15:00 | 2368.60 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-05-24 14:00:00 | 2389.10 | 2024-05-28 09:15:00 | 2368.60 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-05-24 15:15:00 | 2399.00 | 2024-05-28 09:15:00 | 2368.60 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-05-27 13:00:00 | 2403.00 | 2024-05-28 09:15:00 | 2368.60 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest1 | 2024-06-12 09:15:00 | 2534.65 | 2024-06-13 10:15:00 | 2511.05 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest1 | 2024-06-12 11:30:00 | 2532.80 | 2024-06-13 10:15:00 | 2511.05 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest1 | 2024-06-12 13:45:00 | 2531.00 | 2024-06-13 10:15:00 | 2511.05 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-06-13 14:30:00 | 2537.25 | 2024-06-19 09:15:00 | 2508.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-06-14 09:45:00 | 2533.65 | 2024-06-19 09:15:00 | 2508.00 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-06-14 10:15:00 | 2530.80 | 2024-06-19 09:15:00 | 2508.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-06-14 11:45:00 | 2537.55 | 2024-06-19 09:15:00 | 2508.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-06-21 12:30:00 | 2474.50 | 2024-06-21 13:15:00 | 2521.55 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-06-26 09:15:00 | 2451.95 | 2024-06-26 14:15:00 | 2500.05 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-07-05 11:15:00 | 2660.70 | 2024-07-09 13:15:00 | 2730.15 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2024-07-05 12:15:00 | 2669.40 | 2024-07-09 13:15:00 | 2730.15 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-07-08 09:45:00 | 2667.20 | 2024-07-09 13:15:00 | 2730.15 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2024-07-25 09:15:00 | 2599.50 | 2024-07-29 10:15:00 | 2686.00 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2024-07-25 10:00:00 | 2623.00 | 2024-07-29 10:15:00 | 2686.00 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2024-07-26 13:00:00 | 2626.10 | 2024-07-29 10:15:00 | 2686.00 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-07-26 15:00:00 | 2611.00 | 2024-07-29 10:15:00 | 2686.00 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2024-08-02 12:15:00 | 2650.00 | 2024-08-02 14:15:00 | 2675.95 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-08-02 13:15:00 | 2653.25 | 2024-08-02 14:15:00 | 2675.95 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-08-05 09:15:00 | 2575.60 | 2024-08-07 10:15:00 | 2675.00 | STOP_HIT | 1.00 | -3.86% |
| BUY | retest2 | 2024-08-12 14:30:00 | 2728.10 | 2024-08-13 09:15:00 | 2684.90 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-08-23 12:30:00 | 2832.00 | 2024-08-26 10:15:00 | 2808.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-08-26 11:45:00 | 2832.90 | 2024-08-29 10:15:00 | 2819.00 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2024-08-26 12:15:00 | 2833.40 | 2024-08-29 10:15:00 | 2819.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2024-08-26 13:30:00 | 2834.90 | 2024-08-29 10:15:00 | 2819.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-08-27 12:00:00 | 2909.05 | 2024-08-29 10:15:00 | 2819.00 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest2 | 2024-08-28 09:15:00 | 2902.00 | 2024-08-29 10:15:00 | 2819.00 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2024-09-06 11:45:00 | 2877.85 | 2024-09-06 14:15:00 | 2858.25 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2024-09-10 13:15:00 | 2838.45 | 2024-09-11 11:15:00 | 2891.70 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2024-09-18 09:15:00 | 2984.00 | 2024-09-18 13:15:00 | 2948.60 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-10-01 09:15:00 | 3169.80 | 2024-10-03 09:15:00 | 3131.80 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-10-01 10:45:00 | 3163.10 | 2024-10-03 09:15:00 | 3131.80 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-10-04 13:45:00 | 3085.55 | 2024-10-07 10:15:00 | 2931.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-04 13:45:00 | 3085.55 | 2024-10-08 10:15:00 | 2984.10 | STOP_HIT | 0.50 | 3.29% |
| SELL | retest2 | 2024-10-14 11:00:00 | 2987.20 | 2024-10-18 09:15:00 | 2688.48 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-15 09:15:00 | 2972.20 | 2024-10-18 09:15:00 | 2823.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 09:15:00 | 2972.20 | 2024-10-18 09:15:00 | 2699.01 | TARGET_HIT | 0.50 | 9.19% |
| SELL | retest2 | 2024-10-15 12:00:00 | 2998.90 | 2024-10-18 09:15:00 | 2836.75 | PARTIAL | 0.50 | 5.41% |
| SELL | retest2 | 2024-10-15 13:30:00 | 2986.05 | 2024-10-18 09:15:00 | 2725.55 | PARTIAL | 0.50 | 8.72% |
| SELL | retest2 | 2024-10-17 12:15:00 | 2869.00 | 2024-10-18 09:15:00 | 2728.40 | PARTIAL | 0.50 | 4.90% |
| SELL | retest2 | 2024-10-17 14:00:00 | 2872.00 | 2024-10-18 09:15:00 | 2692.06 | PARTIAL | 0.50 | 6.27% |
| SELL | retest2 | 2024-10-15 12:00:00 | 2998.90 | 2024-10-18 11:15:00 | 2948.00 | STOP_HIT | 0.50 | 1.70% |
| SELL | retest2 | 2024-10-15 13:30:00 | 2986.05 | 2024-10-18 11:15:00 | 2948.00 | STOP_HIT | 0.50 | 1.27% |
| SELL | retest2 | 2024-10-17 12:15:00 | 2869.00 | 2024-10-18 11:15:00 | 2948.00 | STOP_HIT | 0.50 | -2.75% |
| SELL | retest2 | 2024-10-17 14:00:00 | 2872.00 | 2024-10-18 11:15:00 | 2948.00 | STOP_HIT | 0.50 | -2.65% |
| SELL | retest2 | 2024-10-18 09:15:00 | 2833.75 | 2024-10-18 11:15:00 | 2948.00 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2024-10-18 10:15:00 | 2862.75 | 2024-10-18 12:15:00 | 2971.20 | STOP_HIT | 1.00 | -3.79% |
| BUY | retest2 | 2024-11-01 18:00:00 | 2835.50 | 2024-11-04 09:15:00 | 2760.00 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2024-11-18 09:30:00 | 2739.75 | 2024-11-18 13:15:00 | 2800.35 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2024-11-18 10:15:00 | 2756.80 | 2024-11-18 13:15:00 | 2800.35 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2024-11-18 11:30:00 | 2757.90 | 2024-11-18 13:15:00 | 2800.35 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-11-18 13:00:00 | 2760.00 | 2024-11-18 13:15:00 | 2800.35 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest1 | 2024-11-28 11:30:00 | 2982.30 | 2024-12-02 12:15:00 | 3122.70 | PARTIAL | 0.50 | 4.71% |
| BUY | retest1 | 2024-11-28 15:15:00 | 2974.00 | 2024-12-03 11:15:00 | 3131.42 | PARTIAL | 0.50 | 5.29% |
| BUY | retest1 | 2024-11-28 11:30:00 | 2982.30 | 2024-12-03 14:15:00 | 3104.75 | STOP_HIT | 0.50 | 4.11% |
| BUY | retest1 | 2024-11-28 15:15:00 | 2974.00 | 2024-12-03 14:15:00 | 3104.75 | STOP_HIT | 0.50 | 4.40% |
| BUY | retest2 | 2024-11-29 11:15:00 | 3006.05 | 2024-12-06 09:15:00 | 3099.10 | STOP_HIT | 1.00 | 3.10% |
| BUY | retest2 | 2024-11-29 12:00:00 | 3016.20 | 2024-12-06 09:15:00 | 3099.10 | STOP_HIT | 1.00 | 2.75% |
| SELL | retest2 | 2024-12-17 11:15:00 | 3136.15 | 2024-12-20 15:15:00 | 2979.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-18 09:45:00 | 3120.05 | 2024-12-20 15:15:00 | 2964.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 11:15:00 | 3136.15 | 2024-12-23 11:15:00 | 3043.60 | STOP_HIT | 0.50 | 2.95% |
| SELL | retest2 | 2024-12-18 09:45:00 | 3120.05 | 2024-12-23 11:15:00 | 3043.60 | STOP_HIT | 0.50 | 2.45% |
| BUY | retest2 | 2024-12-31 14:45:00 | 3232.00 | 2025-01-02 12:15:00 | 3197.40 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-01-02 09:15:00 | 3225.10 | 2025-01-02 12:15:00 | 3197.40 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-01-06 10:15:00 | 3151.80 | 2025-01-16 09:15:00 | 2836.62 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-24 09:30:00 | 2945.75 | 2025-01-28 10:15:00 | 2798.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 15:15:00 | 2931.00 | 2025-01-28 10:15:00 | 2784.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 09:30:00 | 2945.75 | 2025-01-28 11:15:00 | 2870.45 | STOP_HIT | 0.50 | 2.56% |
| SELL | retest2 | 2025-01-24 15:15:00 | 2931.00 | 2025-01-28 11:15:00 | 2870.45 | STOP_HIT | 0.50 | 2.07% |
| BUY | retest1 | 2025-02-04 09:15:00 | 3069.50 | 2025-02-05 09:15:00 | 2994.90 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2025-02-20 15:00:00 | 2745.30 | 2025-02-24 09:15:00 | 2673.75 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2025-02-21 09:15:00 | 2751.20 | 2025-02-24 09:15:00 | 2673.75 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2025-02-21 10:15:00 | 2725.30 | 2025-02-24 09:15:00 | 2673.75 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-02-21 14:45:00 | 2731.20 | 2025-02-24 09:15:00 | 2673.75 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-03-18 09:15:00 | 2664.15 | 2025-03-25 14:15:00 | 2930.57 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-02 09:15:00 | 2913.90 | 2025-04-03 12:15:00 | 2861.45 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-04-03 10:30:00 | 2892.20 | 2025-04-03 12:15:00 | 2861.45 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-04-07 09:15:00 | 2714.60 | 2025-04-07 09:15:00 | 2578.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-07 09:15:00 | 2714.60 | 2025-04-08 09:15:00 | 2709.35 | STOP_HIT | 0.50 | 0.19% |
| BUY | retest2 | 2025-04-23 09:15:00 | 3024.80 | 2025-04-30 11:15:00 | 3327.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-23 11:00:00 | 3012.00 | 2025-04-30 11:15:00 | 3313.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-23 14:45:00 | 3010.30 | 2025-04-30 11:15:00 | 3311.33 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-27 13:00:00 | 3740.50 | 2025-05-28 11:15:00 | 3804.00 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-05-27 14:30:00 | 3739.80 | 2025-05-28 11:15:00 | 3804.00 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-05-28 10:15:00 | 3737.40 | 2025-05-28 11:15:00 | 3804.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-05-28 15:00:00 | 3739.30 | 2025-05-30 09:15:00 | 3782.70 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-05-29 10:15:00 | 3742.40 | 2025-05-30 09:15:00 | 3782.70 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-05-29 11:30:00 | 3741.70 | 2025-05-30 10:15:00 | 3803.00 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-06-10 10:15:00 | 3847.90 | 2025-06-12 12:15:00 | 3775.20 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-06-12 11:00:00 | 3807.40 | 2025-06-12 12:15:00 | 3775.20 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-06-19 11:00:00 | 3579.30 | 2025-06-24 15:15:00 | 3570.00 | STOP_HIT | 1.00 | 0.26% |
| BUY | retest2 | 2025-06-30 15:00:00 | 3684.50 | 2025-07-04 10:15:00 | 3639.90 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-07-01 09:30:00 | 3711.90 | 2025-07-04 10:15:00 | 3639.90 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-07-02 09:30:00 | 3722.10 | 2025-07-04 10:15:00 | 3639.90 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-07-03 09:30:00 | 3720.50 | 2025-07-04 10:15:00 | 3639.90 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest1 | 2025-07-16 09:15:00 | 3934.50 | 2025-07-17 09:15:00 | 3840.40 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest1 | 2025-07-16 13:15:00 | 3897.20 | 2025-07-17 09:15:00 | 3840.40 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest1 | 2025-07-16 14:00:00 | 3896.70 | 2025-07-17 09:15:00 | 3840.40 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-07-17 10:15:00 | 3918.70 | 2025-07-18 09:15:00 | 3790.50 | STOP_HIT | 1.00 | -3.27% |
| BUY | retest2 | 2025-07-17 11:00:00 | 3909.80 | 2025-07-18 09:15:00 | 3790.50 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-08-12 10:15:00 | 3108.20 | 2025-08-13 13:15:00 | 3166.30 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-08-29 09:15:00 | 3088.40 | 2025-09-01 09:15:00 | 3156.60 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-09-08 09:45:00 | 3369.00 | 2025-09-11 12:15:00 | 3326.30 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-09-08 12:45:00 | 3374.90 | 2025-09-11 12:15:00 | 3326.30 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-09-09 14:30:00 | 3372.90 | 2025-09-11 12:15:00 | 3326.30 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-09-10 09:15:00 | 3369.90 | 2025-09-11 12:15:00 | 3326.30 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest1 | 2025-09-18 09:15:00 | 3445.00 | 2025-09-19 09:15:00 | 3395.70 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-09-18 13:45:00 | 3434.20 | 2025-09-19 09:15:00 | 3395.70 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-09-30 13:00:00 | 3406.30 | 2025-10-01 09:15:00 | 3465.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-10-27 09:15:00 | 4162.60 | 2025-10-27 11:15:00 | 4094.90 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-10-30 12:45:00 | 4081.20 | 2025-10-30 13:15:00 | 4083.20 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2025-11-12 11:00:00 | 4104.90 | 2025-11-13 13:15:00 | 4054.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-11-13 09:15:00 | 4105.00 | 2025-11-13 13:15:00 | 4054.00 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-11-18 09:15:00 | 3990.00 | 2025-11-24 14:15:00 | 3790.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 09:15:00 | 3990.00 | 2025-11-25 13:15:00 | 3869.10 | STOP_HIT | 0.50 | 3.03% |
| SELL | retest2 | 2025-12-10 11:45:00 | 3820.00 | 2025-12-15 15:15:00 | 3834.00 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-12-10 13:30:00 | 3822.70 | 2025-12-15 15:15:00 | 3834.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-12-29 10:15:00 | 3840.00 | 2025-12-30 11:15:00 | 3648.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-29 10:15:00 | 3840.00 | 2025-12-31 09:15:00 | 3741.20 | STOP_HIT | 0.50 | 2.57% |
| SELL | retest2 | 2026-01-22 12:45:00 | 3717.60 | 2026-01-23 11:15:00 | 3718.00 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2026-01-22 13:30:00 | 3712.00 | 2026-01-23 11:15:00 | 3718.00 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2026-01-23 09:30:00 | 3707.60 | 2026-01-23 11:15:00 | 3718.00 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2026-01-29 09:15:00 | 3656.00 | 2026-01-30 09:15:00 | 3739.70 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2026-02-06 13:00:00 | 3920.40 | 2026-02-12 14:15:00 | 3995.30 | STOP_HIT | 1.00 | 1.91% |
| SELL | retest2 | 2026-02-19 12:00:00 | 3862.80 | 2026-02-27 09:15:00 | 3669.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 12:30:00 | 3849.50 | 2026-02-27 09:15:00 | 3657.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 13:45:00 | 3860.00 | 2026-02-27 09:15:00 | 3667.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 12:00:00 | 3862.80 | 2026-03-02 09:15:00 | 3476.52 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-19 12:30:00 | 3849.50 | 2026-03-02 09:15:00 | 3464.55 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-19 13:45:00 | 3860.00 | 2026-03-02 09:15:00 | 3474.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-02 09:15:00 | 3287.00 | 2026-04-07 11:15:00 | 3379.10 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2026-04-02 15:15:00 | 3360.00 | 2026-04-07 11:15:00 | 3379.10 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2026-04-07 11:00:00 | 3373.30 | 2026-04-07 11:15:00 | 3379.10 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2026-04-13 10:15:00 | 3565.50 | 2026-04-23 10:15:00 | 3728.60 | STOP_HIT | 1.00 | 4.57% |
