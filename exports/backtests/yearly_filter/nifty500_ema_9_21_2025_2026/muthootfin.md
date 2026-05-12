# Muthoot Finance Ltd. (MUTHOOTFIN)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 3535.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 83 |
| ALERT1 | 63 |
| ALERT2 | 60 |
| ALERT2_SKIP | 34 |
| ALERT3 | 155 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 64 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 64 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 56
- **Target hits / Stop hits / Partials:** 1 / 63 / 0
- **Avg / median % per leg:** -0.81% / -1.06%
- **Sum % (uncompounded):** -51.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 33 | 8 | 24.2% | 1 | 32 | 0 | -0.24% | -7.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 33 | 8 | 24.2% | 1 | 32 | 0 | -0.24% | -7.9% |
| SELL (all) | 31 | 0 | 0.0% | 0 | 31 | 0 | -1.41% | -43.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 31 | 0 | 0.0% | 0 | 31 | 0 | -1.41% | -43.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 64 | 8 | 12.5% | 1 | 63 | 0 | -0.81% | -51.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 2297.30 | 2254.56 | 2251.26 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 09:15:00 | 2223.30 | 2251.66 | 2251.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 12:15:00 | 2211.40 | 2233.98 | 2242.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-14 09:15:00 | 2255.90 | 2231.52 | 2238.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 2255.90 | 2231.52 | 2238.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 2255.90 | 2231.52 | 2238.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-14 09:30:00 | 2251.60 | 2231.52 | 2238.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 10:15:00 | 2248.00 | 2234.82 | 2238.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-14 12:30:00 | 2238.10 | 2241.36 | 2241.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-14 13:15:00 | 2255.10 | 2244.11 | 2242.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 13:15:00 | 2255.10 | 2244.11 | 2242.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 15:15:00 | 2266.00 | 2250.26 | 2245.82 | Break + close above crossover candle high |

### Cycle 4 — SELL (started 2025-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 09:15:00 | 2148.00 | 2229.81 | 2236.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-15 10:15:00 | 2131.00 | 2210.05 | 2227.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 2065.50 | 2052.68 | 2077.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 10:00:00 | 2065.50 | 2052.68 | 2077.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 2072.40 | 2058.24 | 2073.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:30:00 | 2067.50 | 2058.24 | 2073.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 13:15:00 | 2082.50 | 2063.09 | 2074.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 14:00:00 | 2082.50 | 2063.09 | 2074.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 2090.00 | 2068.47 | 2075.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 14:45:00 | 2087.40 | 2068.47 | 2075.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 2092.00 | 2078.80 | 2079.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:45:00 | 2101.00 | 2078.80 | 2079.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 11:15:00 | 2097.70 | 2082.58 | 2080.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 2103.80 | 2092.21 | 2088.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 15:15:00 | 2096.00 | 2096.26 | 2092.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 15:15:00 | 2096.00 | 2096.26 | 2092.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 2096.00 | 2096.26 | 2092.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 2102.20 | 2096.26 | 2092.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 2081.40 | 2093.29 | 2091.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 2081.40 | 2093.29 | 2091.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 10:15:00 | 2072.00 | 2089.03 | 2090.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 12:15:00 | 2069.00 | 2082.14 | 2086.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 09:15:00 | 2075.20 | 2073.38 | 2080.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-28 10:00:00 | 2075.20 | 2073.38 | 2080.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 2091.70 | 2077.04 | 2081.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:00:00 | 2091.70 | 2077.04 | 2081.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 2097.70 | 2081.18 | 2082.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:00:00 | 2097.70 | 2081.18 | 2082.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 12:15:00 | 2099.10 | 2084.76 | 2084.28 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 09:15:00 | 2064.00 | 2084.64 | 2084.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 13:15:00 | 2061.50 | 2072.56 | 2078.44 | Break + close below crossover candle low |

### Cycle 9 — BUY (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 09:15:00 | 2150.20 | 2085.65 | 2082.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 10:15:00 | 2204.80 | 2109.48 | 2093.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 2540.60 | 2542.62 | 2494.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 09:30:00 | 2526.90 | 2542.62 | 2494.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 2549.00 | 2558.63 | 2540.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:15:00 | 2575.00 | 2558.63 | 2540.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 2568.00 | 2560.51 | 2542.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 12:00:00 | 2587.50 | 2566.32 | 2548.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 11:15:00 | 2598.90 | 2622.93 | 2625.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 11:15:00 | 2598.90 | 2622.93 | 2625.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 12:15:00 | 2589.10 | 2616.16 | 2622.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 14:15:00 | 2602.50 | 2602.07 | 2609.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-23 14:45:00 | 2596.00 | 2602.07 | 2609.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 2609.70 | 2603.60 | 2609.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:15:00 | 2586.90 | 2603.60 | 2609.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 2553.20 | 2593.52 | 2604.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 10:15:00 | 2550.00 | 2593.52 | 2604.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 10:00:00 | 2548.80 | 2562.16 | 2579.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 12:30:00 | 2550.00 | 2555.49 | 2571.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 13:30:00 | 2551.50 | 2555.49 | 2569.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 2574.00 | 2559.19 | 2570.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 15:00:00 | 2574.00 | 2559.19 | 2570.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 2590.00 | 2565.35 | 2572.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:15:00 | 2590.30 | 2565.35 | 2572.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 2560.80 | 2564.44 | 2571.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-26 10:45:00 | 2557.50 | 2560.53 | 2568.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 2541.90 | 2569.27 | 2570.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 11:00:00 | 2553.10 | 2560.86 | 2565.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 09:15:00 | 2613.20 | 2575.62 | 2570.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 09:15:00 | 2613.20 | 2575.62 | 2570.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 11:15:00 | 2647.50 | 2596.42 | 2581.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 2606.80 | 2613.36 | 2597.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 10:15:00 | 2605.50 | 2613.36 | 2597.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 2613.90 | 2613.47 | 2598.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:45:00 | 2600.00 | 2613.47 | 2598.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 2619.70 | 2626.28 | 2615.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:00:00 | 2619.70 | 2626.28 | 2615.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 2619.50 | 2624.93 | 2616.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 2608.60 | 2624.93 | 2616.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 2625.60 | 2625.06 | 2617.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 15:15:00 | 2611.30 | 2625.06 | 2617.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 2611.30 | 2622.31 | 2616.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 09:15:00 | 2636.90 | 2622.31 | 2616.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 15:00:00 | 2635.30 | 2634.62 | 2626.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:45:00 | 2634.00 | 2635.00 | 2628.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 11:30:00 | 2635.10 | 2649.91 | 2648.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 13:15:00 | 2636.70 | 2645.68 | 2646.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 13:15:00 | 2636.70 | 2645.68 | 2646.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 11:15:00 | 2634.90 | 2642.65 | 2644.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 12:15:00 | 2645.00 | 2643.12 | 2644.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 12:15:00 | 2645.00 | 2643.12 | 2644.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 2645.00 | 2643.12 | 2644.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:00:00 | 2645.00 | 2643.12 | 2644.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 2646.00 | 2643.69 | 2644.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:45:00 | 2649.90 | 2643.69 | 2644.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 2645.50 | 2644.06 | 2644.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 15:00:00 | 2645.50 | 2644.06 | 2644.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 15:15:00 | 2631.50 | 2641.54 | 2643.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:15:00 | 2648.20 | 2641.54 | 2643.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 2646.50 | 2642.54 | 2643.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 11:00:00 | 2639.30 | 2641.89 | 2643.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 2626.80 | 2640.46 | 2641.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 09:15:00 | 2627.60 | 2629.49 | 2632.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 2661.80 | 2635.95 | 2635.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 09:15:00 | 2661.80 | 2635.95 | 2635.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 10:15:00 | 2670.50 | 2642.86 | 2638.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 10:15:00 | 2655.90 | 2658.22 | 2650.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 11:15:00 | 2657.10 | 2658.22 | 2650.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 2650.30 | 2656.23 | 2651.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:00:00 | 2650.30 | 2656.23 | 2651.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 2646.70 | 2654.33 | 2650.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:00:00 | 2646.70 | 2654.33 | 2650.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 2657.80 | 2655.02 | 2651.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:45:00 | 2641.70 | 2655.02 | 2651.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 2645.00 | 2653.02 | 2650.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:15:00 | 2645.30 | 2653.02 | 2650.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 2645.40 | 2651.49 | 2650.27 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2025-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 11:15:00 | 2641.10 | 2649.00 | 2649.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 09:15:00 | 2631.30 | 2644.07 | 2646.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 11:15:00 | 2650.50 | 2644.78 | 2646.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 11:15:00 | 2650.50 | 2644.78 | 2646.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 2650.50 | 2644.78 | 2646.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:00:00 | 2650.50 | 2644.78 | 2646.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 2647.30 | 2645.29 | 2646.49 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 14:15:00 | 2659.00 | 2649.07 | 2648.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 09:15:00 | 2665.30 | 2656.71 | 2653.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 09:15:00 | 2663.60 | 2672.67 | 2665.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 2663.60 | 2672.67 | 2665.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 2663.60 | 2672.67 | 2665.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:30:00 | 2657.90 | 2672.67 | 2665.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 2659.80 | 2670.10 | 2664.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 11:30:00 | 2673.40 | 2670.30 | 2665.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 10:00:00 | 2670.50 | 2674.46 | 2670.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 10:30:00 | 2674.50 | 2675.31 | 2670.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 14:15:00 | 2663.00 | 2675.31 | 2675.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 14:15:00 | 2663.00 | 2675.31 | 2675.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 12:15:00 | 2638.60 | 2655.80 | 2662.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 14:15:00 | 2634.00 | 2615.90 | 2625.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 14:15:00 | 2634.00 | 2615.90 | 2625.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 2634.00 | 2615.90 | 2625.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 15:00:00 | 2634.00 | 2615.90 | 2625.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 2627.00 | 2618.12 | 2626.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:30:00 | 2625.30 | 2614.89 | 2623.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 2612.60 | 2613.68 | 2620.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 12:45:00 | 2619.90 | 2613.68 | 2620.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 2619.20 | 2614.78 | 2620.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 14:00:00 | 2619.20 | 2614.78 | 2620.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 2600.10 | 2612.11 | 2618.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 15:15:00 | 2582.50 | 2605.28 | 2612.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 09:15:00 | 2637.00 | 2607.98 | 2612.12 | SL hit (close>static) qty=1.00 sl=2630.90 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 11:15:00 | 2638.80 | 2618.60 | 2616.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 12:15:00 | 2649.20 | 2624.72 | 2619.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 09:15:00 | 2628.00 | 2641.47 | 2630.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 09:15:00 | 2628.00 | 2641.47 | 2630.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 2628.00 | 2641.47 | 2630.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:00:00 | 2628.00 | 2641.47 | 2630.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 2642.60 | 2641.70 | 2631.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 15:00:00 | 2644.30 | 2638.59 | 2633.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 10:15:00 | 2622.90 | 2634.04 | 2632.29 | SL hit (close<static) qty=1.00 sl=2623.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 15:15:00 | 2616.00 | 2630.73 | 2631.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 2603.30 | 2623.96 | 2628.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 2633.60 | 2620.31 | 2624.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 2633.60 | 2620.31 | 2624.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 2633.60 | 2620.31 | 2624.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 2633.60 | 2620.31 | 2624.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 2636.10 | 2623.47 | 2625.59 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 10:15:00 | 2644.10 | 2629.60 | 2628.13 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 2600.10 | 2626.90 | 2628.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 09:15:00 | 2570.00 | 2615.52 | 2622.97 | Break + close below crossover candle low |

### Cycle 21 — BUY (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 09:15:00 | 2786.10 | 2569.59 | 2563.62 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 11:15:00 | 2680.00 | 2714.14 | 2714.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 15:15:00 | 2666.00 | 2682.20 | 2692.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 09:15:00 | 2690.30 | 2683.82 | 2692.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 09:15:00 | 2690.30 | 2683.82 | 2692.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 2690.30 | 2683.82 | 2692.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:00:00 | 2690.30 | 2683.82 | 2692.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 2685.70 | 2684.74 | 2691.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:45:00 | 2690.90 | 2684.74 | 2691.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 2692.70 | 2686.33 | 2691.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:45:00 | 2691.80 | 2686.33 | 2691.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 2686.70 | 2686.40 | 2691.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 13:30:00 | 2694.80 | 2686.40 | 2691.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 2695.90 | 2685.27 | 2689.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 2668.50 | 2688.59 | 2689.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 14:00:00 | 2676.30 | 2686.63 | 2688.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 14:30:00 | 2675.30 | 2683.25 | 2686.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 11:15:00 | 2675.60 | 2682.21 | 2685.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 2674.40 | 2680.64 | 2684.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 13:00:00 | 2665.90 | 2677.70 | 2682.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 13:30:00 | 2670.10 | 2675.68 | 2681.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:15:00 | 2670.40 | 2658.02 | 2668.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 10:15:00 | 2706.90 | 2665.21 | 2665.88 | SL hit (close>static) qty=1.00 sl=2685.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 2714.10 | 2674.99 | 2670.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 2747.10 | 2706.79 | 2689.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 12:15:00 | 2712.00 | 2713.59 | 2697.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 13:00:00 | 2712.00 | 2713.59 | 2697.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 2693.90 | 2709.65 | 2696.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 2693.90 | 2709.65 | 2696.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 2699.70 | 2707.66 | 2697.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:30:00 | 2693.70 | 2707.66 | 2697.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 2697.00 | 2705.53 | 2697.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 2754.70 | 2705.53 | 2697.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 13:15:00 | 2879.40 | 2892.77 | 2893.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 2879.40 | 2892.77 | 2893.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 14:15:00 | 2875.00 | 2889.22 | 2891.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 09:15:00 | 2898.40 | 2889.90 | 2891.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 2898.40 | 2889.90 | 2891.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 2898.40 | 2889.90 | 2891.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:00:00 | 2898.40 | 2889.90 | 2891.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 2890.70 | 2890.06 | 2891.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 11:45:00 | 2885.80 | 2889.65 | 2891.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 12:15:00 | 2915.90 | 2894.90 | 2893.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 12:15:00 | 2915.90 | 2894.90 | 2893.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 14:15:00 | 2926.20 | 2903.93 | 2897.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 10:15:00 | 2949.00 | 2964.79 | 2948.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 10:15:00 | 2949.00 | 2964.79 | 2948.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 2949.00 | 2964.79 | 2948.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:00:00 | 2949.00 | 2964.79 | 2948.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 2958.20 | 2963.47 | 2948.95 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 10:15:00 | 2923.10 | 2940.18 | 2942.45 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 15:15:00 | 2955.00 | 2943.66 | 2942.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 09:15:00 | 2963.30 | 2947.59 | 2944.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 10:15:00 | 2946.00 | 2947.27 | 2944.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-19 11:00:00 | 2946.00 | 2947.27 | 2944.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 2927.20 | 2943.26 | 2943.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:00:00 | 2927.20 | 2943.26 | 2943.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 12:15:00 | 2926.40 | 2939.88 | 2941.61 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 3008.80 | 2952.89 | 2946.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 13:15:00 | 3022.50 | 2991.46 | 2969.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 09:15:00 | 3048.10 | 3067.79 | 3048.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 3048.10 | 3067.79 | 3048.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 3048.10 | 3067.79 | 3048.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:45:00 | 3054.20 | 3067.79 | 3048.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 3047.00 | 3063.63 | 3047.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:45:00 | 3042.70 | 3063.63 | 3047.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 3044.70 | 3059.85 | 3047.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:15:00 | 3038.30 | 3059.85 | 3047.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 12:15:00 | 3066.10 | 3061.10 | 3049.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:30:00 | 3058.60 | 3061.10 | 3049.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 3048.80 | 3059.76 | 3050.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:00:00 | 3048.80 | 3059.76 | 3050.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 3045.00 | 3056.81 | 3050.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:15:00 | 3040.20 | 3056.81 | 3050.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 3047.30 | 3054.91 | 3050.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:30:00 | 3061.60 | 3054.91 | 3050.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 3049.90 | 3053.90 | 3050.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:30:00 | 3048.90 | 3053.90 | 3050.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 3025.00 | 3048.12 | 3047.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:30:00 | 3025.20 | 3048.12 | 3047.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 12:15:00 | 3043.80 | 3047.26 | 3047.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 13:15:00 | 3018.70 | 3041.55 | 3044.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 3036.80 | 3034.63 | 3040.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 3036.80 | 3034.63 | 3040.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 3036.80 | 3034.63 | 3040.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:30:00 | 3034.90 | 3034.63 | 3040.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 3033.70 | 3034.44 | 3039.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:45:00 | 3033.00 | 3034.44 | 3039.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 3049.20 | 3036.32 | 3039.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:00:00 | 3049.20 | 3036.32 | 3039.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-09-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 13:15:00 | 3072.70 | 3043.59 | 3042.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 09:15:00 | 3077.80 | 3055.31 | 3048.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-30 13:15:00 | 3061.00 | 3061.88 | 3054.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-30 14:00:00 | 3061.00 | 3061.88 | 3054.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 3076.20 | 3064.74 | 3056.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 09:30:00 | 3094.40 | 3073.74 | 3062.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 10:15:00 | 3181.80 | 3227.54 | 3228.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 10:15:00 | 3181.80 | 3227.54 | 3228.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 12:15:00 | 3163.80 | 3205.25 | 3217.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 10:15:00 | 3191.90 | 3183.17 | 3199.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-13 10:45:00 | 3190.40 | 3183.17 | 3199.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 3195.30 | 3185.60 | 3199.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:45:00 | 3198.30 | 3185.60 | 3199.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 3200.10 | 3188.50 | 3199.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:00:00 | 3200.10 | 3188.50 | 3199.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 3208.40 | 3192.48 | 3200.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:45:00 | 3210.00 | 3192.48 | 3200.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 3215.20 | 3197.02 | 3201.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 15:00:00 | 3215.20 | 3197.02 | 3201.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 09:15:00 | 3264.10 | 3213.80 | 3208.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 3351.90 | 3280.45 | 3259.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 09:15:00 | 3293.20 | 3319.29 | 3296.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 09:15:00 | 3293.20 | 3319.29 | 3296.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 3293.20 | 3319.29 | 3296.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 14:30:00 | 3337.90 | 3312.42 | 3300.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 3278.00 | 3306.91 | 3300.05 | SL hit (close<static) qty=1.00 sl=3282.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 09:15:00 | 3167.60 | 3273.46 | 3285.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 11:15:00 | 3118.40 | 3220.66 | 3258.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 10:15:00 | 3207.20 | 3192.06 | 3223.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-24 11:00:00 | 3207.20 | 3192.06 | 3223.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 3173.40 | 3153.96 | 3171.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 3173.40 | 3153.96 | 3171.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 3201.00 | 3163.36 | 3173.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:45:00 | 3212.00 | 3163.36 | 3173.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 3200.10 | 3170.71 | 3176.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:15:00 | 3173.20 | 3170.71 | 3176.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 09:15:00 | 3182.00 | 3175.34 | 3174.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 09:15:00 | 3182.00 | 3175.34 | 3174.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 14:15:00 | 3192.00 | 3179.99 | 3177.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 14:15:00 | 3176.00 | 3193.91 | 3188.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 14:15:00 | 3176.00 | 3193.91 | 3188.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 3176.00 | 3193.91 | 3188.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 3176.00 | 3193.91 | 3188.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 3191.00 | 3193.33 | 3188.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 3175.40 | 3193.33 | 3188.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 3194.30 | 3193.52 | 3189.05 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 3182.00 | 3187.36 | 3187.99 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 12:15:00 | 3199.30 | 3189.75 | 3189.02 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 14:15:00 | 3183.00 | 3188.44 | 3188.55 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 15:15:00 | 3191.00 | 3188.95 | 3188.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-06 09:15:00 | 3210.00 | 3193.16 | 3190.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 10:15:00 | 3177.80 | 3190.09 | 3189.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 10:15:00 | 3177.80 | 3190.09 | 3189.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 3177.80 | 3190.09 | 3189.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:00:00 | 3177.80 | 3190.09 | 3189.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 3181.80 | 3188.43 | 3188.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 3169.00 | 3181.83 | 3185.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 3190.40 | 3183.54 | 3185.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 10:15:00 | 3190.40 | 3183.54 | 3185.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 3190.40 | 3183.54 | 3185.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:45:00 | 3188.00 | 3183.54 | 3185.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 3183.90 | 3183.61 | 3185.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:30:00 | 3185.50 | 3183.61 | 3185.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 3198.80 | 3186.65 | 3186.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:45:00 | 3203.30 | 3186.65 | 3186.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 13:15:00 | 3222.00 | 3193.72 | 3190.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 09:15:00 | 3243.00 | 3212.43 | 3200.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 09:15:00 | 3309.90 | 3338.94 | 3300.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 10:00:00 | 3309.90 | 3338.94 | 3300.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 3357.50 | 3333.44 | 3314.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 10:30:00 | 3382.10 | 3343.51 | 3320.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-14 09:15:00 | 3720.31 | 3442.68 | 3380.42 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 3663.30 | 3684.63 | 3686.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 3635.00 | 3666.03 | 3677.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 3656.90 | 3634.08 | 3649.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 3656.90 | 3634.08 | 3649.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 3656.90 | 3634.08 | 3649.41 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 13:15:00 | 3684.00 | 3660.05 | 3658.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 15:15:00 | 3697.90 | 3670.45 | 3663.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 12:15:00 | 3739.30 | 3750.65 | 3731.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 13:00:00 | 3739.30 | 3750.65 | 3731.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 3746.30 | 3749.78 | 3732.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 14:15:00 | 3752.90 | 3749.78 | 3732.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 3785.80 | 3746.70 | 3734.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 3727.60 | 3776.25 | 3774.90 | SL hit (close<static) qty=1.00 sl=3731.50 alert=retest2 |

### Cycle 44 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 3717.10 | 3764.42 | 3769.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 14:15:00 | 3705.90 | 3732.01 | 3748.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 09:15:00 | 3741.90 | 3728.87 | 3743.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 3741.90 | 3728.87 | 3743.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 3741.90 | 3728.87 | 3743.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 3741.90 | 3728.87 | 3743.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 3756.60 | 3734.41 | 3744.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:00:00 | 3756.60 | 3734.41 | 3744.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 3760.80 | 3739.69 | 3746.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:30:00 | 3773.40 | 3739.69 | 3746.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 3787.00 | 3754.91 | 3752.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 3803.80 | 3764.69 | 3757.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 10:15:00 | 3773.00 | 3773.66 | 3763.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 11:00:00 | 3773.00 | 3773.66 | 3763.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 3768.40 | 3772.61 | 3764.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:15:00 | 3761.60 | 3772.61 | 3764.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 3775.90 | 3773.27 | 3765.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:30:00 | 3767.80 | 3773.27 | 3765.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 3773.50 | 3773.32 | 3765.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:30:00 | 3774.10 | 3773.32 | 3765.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 3770.90 | 3772.83 | 3766.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:30:00 | 3772.90 | 3772.83 | 3766.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 15:15:00 | 3763.10 | 3770.89 | 3766.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:15:00 | 3733.10 | 3770.89 | 3766.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 3737.50 | 3764.21 | 3763.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:00:00 | 3737.50 | 3764.21 | 3763.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 10:15:00 | 3736.60 | 3758.69 | 3761.08 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 09:15:00 | 3770.40 | 3755.75 | 3755.70 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 13:15:00 | 3734.40 | 3751.97 | 3754.21 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 3784.60 | 3759.63 | 3756.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 11:15:00 | 3795.00 | 3766.70 | 3760.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 3824.10 | 3839.90 | 3816.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:00:00 | 3824.10 | 3839.90 | 3816.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 3836.70 | 3839.49 | 3827.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 3836.70 | 3839.49 | 3827.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 3835.80 | 3838.75 | 3828.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:30:00 | 3831.90 | 3838.75 | 3828.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 3800.90 | 3831.18 | 3825.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:00:00 | 3800.90 | 3831.18 | 3825.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 3778.90 | 3820.72 | 3821.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 14:15:00 | 3766.10 | 3802.68 | 3812.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 13:15:00 | 3760.20 | 3753.87 | 3769.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 14:00:00 | 3760.20 | 3753.87 | 3769.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 3780.10 | 3759.12 | 3770.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 3780.10 | 3759.12 | 3770.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 3767.70 | 3760.84 | 3770.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 09:30:00 | 3764.80 | 3763.17 | 3770.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 10:30:00 | 3767.00 | 3762.57 | 3769.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 11:00:00 | 3760.20 | 3762.57 | 3769.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 14:15:00 | 3789.80 | 3775.62 | 3774.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-12-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 14:15:00 | 3789.80 | 3775.62 | 3774.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 12:15:00 | 3801.00 | 3787.91 | 3781.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 13:15:00 | 3837.10 | 3840.28 | 3816.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 14:00:00 | 3837.10 | 3840.28 | 3816.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 3809.40 | 3834.10 | 3815.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 3809.40 | 3834.10 | 3815.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 3796.70 | 3826.62 | 3814.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 3842.90 | 3826.62 | 3814.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 11:15:00 | 3814.90 | 3822.07 | 3814.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 13:45:00 | 3815.40 | 3818.30 | 3814.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 3781.70 | 3806.02 | 3809.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 3781.70 | 3806.02 | 3809.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 11:15:00 | 3767.30 | 3794.56 | 3803.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 12:15:00 | 3767.70 | 3760.10 | 3775.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 12:15:00 | 3767.70 | 3760.10 | 3775.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 3767.70 | 3760.10 | 3775.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:45:00 | 3772.90 | 3760.10 | 3775.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 3798.20 | 3767.72 | 3777.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 13:30:00 | 3788.30 | 3767.72 | 3777.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 3801.30 | 3774.44 | 3779.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 3801.30 | 3774.44 | 3779.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 3794.70 | 3778.49 | 3781.06 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 3847.20 | 3792.23 | 3787.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 3852.40 | 3827.58 | 3813.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 13:15:00 | 3817.70 | 3839.05 | 3825.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 13:15:00 | 3817.70 | 3839.05 | 3825.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 3817.70 | 3839.05 | 3825.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:00:00 | 3817.70 | 3839.05 | 3825.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 3820.10 | 3835.26 | 3824.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:45:00 | 3802.80 | 3835.26 | 3824.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 15:15:00 | 3827.00 | 3833.61 | 3824.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 3889.10 | 3833.61 | 3824.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 11:15:00 | 3840.10 | 3901.50 | 3909.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 3840.10 | 3901.50 | 3909.70 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 15:15:00 | 3911.00 | 3878.83 | 3874.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 10:15:00 | 3940.00 | 3895.89 | 3883.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 09:15:00 | 3908.60 | 3909.58 | 3897.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 3908.60 | 3909.58 | 3897.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 3908.60 | 3909.58 | 3897.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 12:15:00 | 3943.80 | 3917.25 | 3902.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 14:30:00 | 3940.40 | 3930.76 | 3913.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 15:00:00 | 3940.00 | 3930.76 | 3913.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 10:15:00 | 3950.10 | 3932.37 | 3917.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 3939.60 | 3938.57 | 3926.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:30:00 | 3936.00 | 3938.57 | 3926.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 3926.30 | 3936.12 | 3926.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 3963.60 | 3936.12 | 3926.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 12:15:00 | 3945.00 | 3935.68 | 3928.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 13:00:00 | 3946.70 | 3937.89 | 3930.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 11:15:00 | 3903.30 | 3930.06 | 3930.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 3903.30 | 3930.06 | 3930.58 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 09:15:00 | 3953.30 | 3930.72 | 3929.71 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 14:15:00 | 3905.80 | 3930.89 | 3931.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-22 09:15:00 | 3825.00 | 3906.40 | 3919.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 13:15:00 | 3870.60 | 3865.63 | 3892.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 14:00:00 | 3870.60 | 3865.63 | 3892.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 3906.50 | 3872.16 | 3888.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 12:00:00 | 3876.00 | 3879.94 | 3889.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 10:15:00 | 3946.30 | 3884.99 | 3877.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 3946.30 | 3884.99 | 3877.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 3956.00 | 3918.73 | 3897.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 3965.90 | 4034.81 | 3988.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 3965.90 | 4034.81 | 3988.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 3965.90 | 4034.81 | 3988.78 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 13:15:00 | 3863.80 | 3953.24 | 3961.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 14:15:00 | 3828.60 | 3928.31 | 3949.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 3539.50 | 3515.63 | 3627.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 3539.50 | 3515.63 | 3627.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 3603.80 | 3541.96 | 3620.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 12:15:00 | 3602.10 | 3569.23 | 3620.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 13:30:00 | 3602.80 | 3589.82 | 3621.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 11:15:00 | 3684.00 | 3639.75 | 3636.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 11:15:00 | 3684.00 | 3639.75 | 3636.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 12:15:00 | 3711.80 | 3654.16 | 3643.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 3585.30 | 3668.56 | 3657.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 3585.30 | 3668.56 | 3657.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 3585.30 | 3668.56 | 3657.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 3583.70 | 3668.56 | 3657.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 3560.40 | 3646.93 | 3648.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 3491.60 | 3581.22 | 3611.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 11:15:00 | 3582.90 | 3568.01 | 3599.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 12:00:00 | 3582.90 | 3568.01 | 3599.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 3626.10 | 3579.63 | 3601.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:30:00 | 3642.70 | 3579.63 | 3601.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 3648.20 | 3593.35 | 3605.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 14:00:00 | 3648.20 | 3593.35 | 3605.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 3704.50 | 3615.58 | 3614.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 3785.00 | 3663.77 | 3637.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 3622.10 | 3919.55 | 3898.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 3622.10 | 3919.55 | 3898.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 3622.10 | 3919.55 | 3898.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:45:00 | 3592.20 | 3919.55 | 3898.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 3585.40 | 3852.72 | 3869.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 3576.10 | 3702.76 | 3784.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 11:15:00 | 3509.90 | 3508.09 | 3595.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 12:00:00 | 3509.90 | 3508.09 | 3595.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 3450.90 | 3420.85 | 3450.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:00:00 | 3450.90 | 3420.85 | 3450.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 3462.00 | 3429.08 | 3451.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 15:00:00 | 3462.00 | 3429.08 | 3451.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 3461.00 | 3435.46 | 3452.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:15:00 | 3505.40 | 3435.46 | 3452.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 3472.40 | 3461.09 | 3461.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:45:00 | 3471.40 | 3461.09 | 3461.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 13:15:00 | 3471.30 | 3463.13 | 3462.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 10:15:00 | 3483.20 | 3469.29 | 3465.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 12:15:00 | 3475.90 | 3487.33 | 3479.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 12:15:00 | 3475.90 | 3487.33 | 3479.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 3475.90 | 3487.33 | 3479.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 12:45:00 | 3473.10 | 3487.33 | 3479.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 3493.20 | 3488.50 | 3480.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 14:15:00 | 3496.40 | 3488.50 | 3480.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 11:15:00 | 3444.40 | 3480.34 | 3480.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 11:15:00 | 3444.40 | 3480.34 | 3480.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 12:15:00 | 3426.10 | 3469.49 | 3475.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 09:15:00 | 3424.10 | 3386.61 | 3413.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 3424.10 | 3386.61 | 3413.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 3424.10 | 3386.61 | 3413.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:30:00 | 3427.90 | 3386.61 | 3413.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 10:15:00 | 3433.00 | 3395.89 | 3415.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 11:00:00 | 3433.00 | 3395.89 | 3415.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 11:15:00 | 3417.50 | 3400.21 | 3415.54 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2026-03-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 14:15:00 | 3472.10 | 3428.74 | 3425.76 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 3373.50 | 3423.57 | 3424.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 10:15:00 | 3345.00 | 3407.86 | 3417.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 15:15:00 | 3306.90 | 3301.36 | 3338.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 09:15:00 | 3318.40 | 3301.36 | 3338.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 15:15:00 | 3249.40 | 3223.42 | 3253.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:15:00 | 3287.20 | 3223.42 | 3253.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 3296.80 | 3238.10 | 3257.36 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 3280.30 | 3267.53 | 3266.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 3297.00 | 3273.43 | 3269.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 09:15:00 | 3248.10 | 3268.36 | 3267.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 09:15:00 | 3248.10 | 3268.36 | 3267.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 3248.10 | 3268.36 | 3267.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:45:00 | 3249.70 | 3268.36 | 3267.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 10:15:00 | 3208.50 | 3256.39 | 3262.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 11:15:00 | 3187.40 | 3242.59 | 3255.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 3189.60 | 3188.05 | 3214.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 12:00:00 | 3189.60 | 3188.05 | 3214.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 3244.10 | 3202.18 | 3216.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:30:00 | 3240.70 | 3202.18 | 3216.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 3245.50 | 3210.84 | 3219.51 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 09:15:00 | 3280.00 | 3229.18 | 3226.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-13 10:15:00 | 3314.50 | 3246.24 | 3234.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 3353.80 | 3408.09 | 3386.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 3353.80 | 3408.09 | 3386.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 3353.80 | 3408.09 | 3386.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:15:00 | 3353.70 | 3408.09 | 3386.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 3360.00 | 3398.47 | 3383.89 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 3314.40 | 3375.15 | 3375.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 3310.10 | 3362.14 | 3369.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 14:15:00 | 3317.10 | 3312.78 | 3333.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 15:00:00 | 3317.10 | 3312.78 | 3333.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 3135.40 | 3122.04 | 3164.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 3290.60 | 3122.04 | 3164.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 3293.10 | 3156.25 | 3175.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 3293.10 | 3156.25 | 3175.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 3263.70 | 3195.81 | 3191.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 3290.90 | 3214.83 | 3200.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 10:15:00 | 3236.80 | 3259.62 | 3233.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 10:15:00 | 3236.80 | 3259.62 | 3233.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 3236.80 | 3259.62 | 3233.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 3249.20 | 3259.62 | 3233.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 3210.00 | 3249.70 | 3230.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 3210.00 | 3249.70 | 3230.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 3239.50 | 3247.66 | 3231.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 13:30:00 | 3242.30 | 3246.13 | 3232.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 15:00:00 | 3260.70 | 3249.04 | 3235.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 12:15:00 | 3177.60 | 3220.16 | 3225.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 12:15:00 | 3177.60 | 3220.16 | 3225.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 3166.90 | 3203.47 | 3216.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 3253.30 | 3211.28 | 3217.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 3253.30 | 3211.28 | 3217.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 3253.30 | 3211.28 | 3217.89 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 3257.40 | 3228.11 | 3224.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 13:15:00 | 3280.00 | 3245.83 | 3233.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 3155.60 | 3230.41 | 3230.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 3155.60 | 3230.41 | 3230.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 3155.60 | 3230.41 | 3230.13 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 3180.80 | 3220.49 | 3225.65 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 3273.50 | 3220.84 | 3216.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 3286.60 | 3233.99 | 3222.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 3249.00 | 3252.18 | 3235.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:00:00 | 3249.00 | 3252.18 | 3235.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 3238.30 | 3247.98 | 3236.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 12:30:00 | 3240.10 | 3247.98 | 3236.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 13:15:00 | 3242.90 | 3246.97 | 3236.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 3421.10 | 3245.24 | 3237.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 11:15:00 | 3502.40 | 3567.46 | 3570.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 11:15:00 | 3502.40 | 3567.46 | 3570.47 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 10:15:00 | 3596.20 | 3559.22 | 3558.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 11:15:00 | 3610.80 | 3569.53 | 3563.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 09:15:00 | 3590.70 | 3595.97 | 3581.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 09:15:00 | 3590.70 | 3595.97 | 3581.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 3590.70 | 3595.97 | 3581.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 13:00:00 | 3640.00 | 3607.37 | 3590.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 14:15:00 | 3572.80 | 3601.89 | 3591.16 | SL hit (close<static) qty=1.00 sl=3577.00 alert=retest2 |

### Cycle 80 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 3546.20 | 3585.70 | 3586.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 3541.10 | 3563.62 | 3573.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 12:15:00 | 3509.40 | 3504.26 | 3524.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 12:45:00 | 3510.50 | 3504.26 | 3524.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 3521.90 | 3507.79 | 3523.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 14:00:00 | 3521.90 | 3507.79 | 3523.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 3498.40 | 3505.91 | 3521.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 15:15:00 | 3490.00 | 3505.91 | 3521.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:15:00 | 3490.20 | 3501.47 | 3515.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 09:30:00 | 3488.50 | 3498.31 | 3506.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 10:30:00 | 3489.70 | 3496.69 | 3504.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 3491.60 | 3447.35 | 3461.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:30:00 | 3502.00 | 3447.35 | 3461.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 3478.60 | 3453.60 | 3463.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:30:00 | 3485.00 | 3453.60 | 3463.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-05-04 14:15:00 | 3491.00 | 3472.34 | 3470.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2026-05-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 14:15:00 | 3491.00 | 3472.34 | 3470.17 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 09:15:00 | 3423.70 | 3466.40 | 3468.08 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 3480.70 | 3462.51 | 3461.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 3491.50 | 3470.99 | 3465.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 3532.50 | 3557.95 | 3529.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 3532.50 | 3557.95 | 3529.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 3532.50 | 3557.95 | 3529.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 3520.10 | 3557.95 | 3529.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 3535.20 | 3553.40 | 3529.63 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-14 12:30:00 | 2238.10 | 2025-05-14 13:15:00 | 2255.10 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-06-13 12:00:00 | 2587.50 | 2025-06-20 11:15:00 | 2598.90 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2025-06-24 10:15:00 | 2550.00 | 2025-06-30 09:15:00 | 2613.20 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-06-25 10:00:00 | 2548.80 | 2025-06-30 09:15:00 | 2613.20 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2025-06-25 12:30:00 | 2550.00 | 2025-06-30 09:15:00 | 2613.20 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-06-25 13:30:00 | 2551.50 | 2025-06-30 09:15:00 | 2613.20 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-06-26 10:45:00 | 2557.50 | 2025-06-30 09:15:00 | 2613.20 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-06-27 09:15:00 | 2541.90 | 2025-06-30 09:15:00 | 2613.20 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2025-06-27 11:00:00 | 2553.10 | 2025-06-30 09:15:00 | 2613.20 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-07-03 09:15:00 | 2636.90 | 2025-07-08 13:15:00 | 2636.70 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2025-07-03 15:00:00 | 2635.30 | 2025-07-08 13:15:00 | 2636.70 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2025-07-04 09:45:00 | 2634.00 | 2025-07-08 13:15:00 | 2636.70 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2025-07-08 11:30:00 | 2635.10 | 2025-07-08 13:15:00 | 2636.70 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-07-10 11:00:00 | 2639.30 | 2025-07-14 09:15:00 | 2661.80 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-07-11 09:15:00 | 2626.80 | 2025-07-14 09:15:00 | 2661.80 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-07-14 09:15:00 | 2627.60 | 2025-07-14 09:15:00 | 2661.80 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-07-22 11:30:00 | 2673.40 | 2025-07-24 14:15:00 | 2663.00 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-07-23 10:00:00 | 2670.50 | 2025-07-24 14:15:00 | 2663.00 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-07-23 10:30:00 | 2674.50 | 2025-07-24 14:15:00 | 2663.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-08-01 15:15:00 | 2582.50 | 2025-08-04 09:15:00 | 2637.00 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-08-05 15:00:00 | 2644.30 | 2025-08-06 10:15:00 | 2622.90 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-08-06 12:15:00 | 2645.20 | 2025-08-06 15:15:00 | 2616.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-08-06 13:15:00 | 2643.60 | 2025-08-06 15:15:00 | 2616.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-08-26 09:15:00 | 2668.50 | 2025-09-01 10:15:00 | 2706.90 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-08-26 14:00:00 | 2676.30 | 2025-09-01 10:15:00 | 2706.90 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-08-26 14:30:00 | 2675.30 | 2025-09-01 10:15:00 | 2706.90 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-08-28 11:15:00 | 2675.60 | 2025-09-01 11:15:00 | 2714.10 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-08-28 13:00:00 | 2665.90 | 2025-09-01 11:15:00 | 2714.10 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-08-28 13:30:00 | 2670.10 | 2025-09-01 11:15:00 | 2714.10 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-08-29 12:15:00 | 2670.40 | 2025-09-01 11:15:00 | 2714.10 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-09-03 09:15:00 | 2754.70 | 2025-09-11 13:15:00 | 2879.40 | STOP_HIT | 1.00 | 4.53% |
| SELL | retest2 | 2025-09-12 11:45:00 | 2885.80 | 2025-09-12 12:15:00 | 2915.90 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-10-01 09:30:00 | 3094.40 | 2025-10-10 10:15:00 | 3181.80 | STOP_HIT | 1.00 | 2.82% |
| BUY | retest2 | 2025-10-20 14:30:00 | 3337.90 | 2025-10-21 13:15:00 | 3278.00 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-10-28 14:15:00 | 3173.20 | 2025-10-30 09:15:00 | 3182.00 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-11-13 10:30:00 | 3382.10 | 2025-11-14 09:15:00 | 3720.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-28 14:15:00 | 3752.90 | 2025-12-03 09:15:00 | 3727.60 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-12-01 09:15:00 | 3785.80 | 2025-12-03 09:15:00 | 3727.60 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-12-22 09:30:00 | 3764.80 | 2025-12-22 14:15:00 | 3789.80 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-12-22 10:30:00 | 3767.00 | 2025-12-22 14:15:00 | 3789.80 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-12-22 11:00:00 | 3760.20 | 2025-12-22 14:15:00 | 3789.80 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-12-26 09:15:00 | 3842.90 | 2025-12-29 09:15:00 | 3781.70 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-12-26 11:15:00 | 3814.90 | 2025-12-29 09:15:00 | 3781.70 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-12-26 13:45:00 | 3815.40 | 2025-12-29 09:15:00 | 3781.70 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2026-01-05 09:15:00 | 3889.10 | 2026-01-08 11:15:00 | 3840.10 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2026-01-14 12:15:00 | 3943.80 | 2026-01-20 11:15:00 | 3903.30 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2026-01-14 14:30:00 | 3940.40 | 2026-01-20 11:15:00 | 3903.30 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2026-01-14 15:00:00 | 3940.00 | 2026-01-20 11:15:00 | 3903.30 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2026-01-16 10:15:00 | 3950.10 | 2026-01-20 11:15:00 | 3903.30 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2026-01-19 09:15:00 | 3963.60 | 2026-01-20 11:15:00 | 3903.30 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2026-01-19 12:15:00 | 3945.00 | 2026-01-20 11:15:00 | 3903.30 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2026-01-19 13:00:00 | 3946.70 | 2026-01-20 11:15:00 | 3903.30 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-01-23 12:00:00 | 3876.00 | 2026-01-28 10:15:00 | 3946.30 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2026-02-03 12:15:00 | 3602.10 | 2026-02-04 11:15:00 | 3684.00 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2026-02-03 13:30:00 | 3602.80 | 2026-02-04 11:15:00 | 3684.00 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2026-02-25 14:15:00 | 3496.40 | 2026-02-26 11:15:00 | 3444.40 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2026-03-27 13:30:00 | 3242.30 | 2026-03-30 12:15:00 | 3177.60 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2026-03-27 15:00:00 | 3260.70 | 2026-03-30 12:15:00 | 3177.60 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2026-04-08 09:15:00 | 3421.10 | 2026-04-17 11:15:00 | 3502.40 | STOP_HIT | 1.00 | 2.38% |
| BUY | retest2 | 2026-04-22 13:00:00 | 3640.00 | 2026-04-22 14:15:00 | 3572.80 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2026-04-27 15:15:00 | 3490.00 | 2026-05-04 14:15:00 | 3491.00 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2026-04-28 11:15:00 | 3490.20 | 2026-05-04 14:15:00 | 3491.00 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2026-04-29 09:30:00 | 3488.50 | 2026-05-04 14:15:00 | 3491.00 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2026-04-29 10:30:00 | 3489.70 | 2026-05-04 14:15:00 | 3491.00 | STOP_HIT | 1.00 | -0.04% |
