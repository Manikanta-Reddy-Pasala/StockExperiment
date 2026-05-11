# Glaxosmithkline Pharmaceuticals Ltd. (GLAXO)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 2480.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 3 |
| ALERT3 | 40 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 34 |
| PARTIAL | 5 |
| TARGET_HIT | 4 |
| STOP_HIT | 30 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 39 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 26
- **Target hits / Stop hits / Partials:** 4 / 30 / 5
- **Avg / median % per leg:** 0.83% / -1.19%
- **Sum % (uncompounded):** 32.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 3 | 33.3% | 3 | 6 | 0 | 2.11% | 19.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 3 | 33.3% | 3 | 6 | 0 | 2.11% | 19.0% |
| SELL (all) | 30 | 10 | 33.3% | 1 | 24 | 5 | 0.45% | 13.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 30 | 10 | 33.3% | 1 | 24 | 5 | 0.45% | 13.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 39 | 13 | 33.3% | 4 | 30 | 5 | 0.83% | 32.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 14:15:00 | 1883.10 | 2059.18 | 2059.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-19 09:15:00 | 1866.20 | 1974.58 | 2008.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-24 09:15:00 | 1994.90 | 1964.87 | 2000.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-24 10:00:00 | 1994.90 | 1964.87 | 2000.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 10:15:00 | 2012.00 | 1965.34 | 2000.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-24 10:30:00 | 2009.00 | 1965.34 | 2000.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 11:15:00 | 2021.60 | 1965.90 | 2000.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-24 11:45:00 | 2035.90 | 1965.90 | 2000.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 2059.45 | 2023.88 | 2024.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 10:15:00 | 2052.15 | 2023.88 | 2024.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 12:15:00 | 2017.75 | 2023.98 | 2024.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 13:15:00 | 2013.40 | 2023.98 | 2024.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 14:15:00 | 2014.05 | 2023.96 | 2024.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 10:00:00 | 2009.00 | 2023.60 | 2024.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 10:30:00 | 2015.00 | 2023.38 | 2024.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 14:15:00 | 1989.95 | 2014.09 | 2019.34 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-16 12:15:00 | 2037.45 | 2013.50 | 2018.91 | SL hit (close>static) qty=1.00 sl=2028.85 alert=retest2 |

### Cycle 2 — BUY (started 2024-05-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 15:15:00 | 2299.50 | 2023.93 | 2023.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-18 09:15:00 | 2369.95 | 2027.37 | 2025.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 13:15:00 | 2538.00 | 2540.44 | 2405.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-05 13:45:00 | 2541.65 | 2540.44 | 2405.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 2810.90 | 2825.06 | 2744.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 14:30:00 | 2805.00 | 2825.06 | 2744.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 2745.50 | 2822.66 | 2746.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:00:00 | 2745.50 | 2822.66 | 2746.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 2740.00 | 2821.83 | 2746.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:45:00 | 2738.95 | 2821.83 | 2746.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 2700.00 | 2820.62 | 2746.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:45:00 | 2697.00 | 2820.62 | 2746.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 2735.50 | 2813.32 | 2745.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 12:00:00 | 2735.50 | 2813.32 | 2745.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 12:15:00 | 2733.85 | 2812.53 | 2745.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 12:45:00 | 2729.55 | 2812.53 | 2745.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 2726.55 | 2789.99 | 2741.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:45:00 | 2725.80 | 2789.99 | 2741.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 2720.90 | 2785.90 | 2741.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:00:00 | 2720.90 | 2785.90 | 2741.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 2745.60 | 2781.48 | 2740.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:45:00 | 2742.80 | 2781.48 | 2740.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 2721.25 | 2780.44 | 2740.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 14:00:00 | 2721.25 | 2780.44 | 2740.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 2747.70 | 2780.12 | 2740.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 10:30:00 | 2758.65 | 2771.88 | 2739.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 14:15:00 | 2760.40 | 2771.46 | 2739.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-07 09:15:00 | 2708.45 | 2770.44 | 2739.77 | SL hit (close<static) qty=1.00 sl=2720.10 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 11:15:00 | 2609.80 | 2723.35 | 2723.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 2574.25 | 2712.99 | 2718.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 10:15:00 | 2712.70 | 2695.49 | 2708.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 10:15:00 | 2712.70 | 2695.49 | 2708.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 10:15:00 | 2712.70 | 2695.49 | 2708.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 11:00:00 | 2712.70 | 2695.49 | 2708.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 11:15:00 | 2730.55 | 2695.83 | 2708.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 12:15:00 | 2739.00 | 2695.83 | 2708.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 2718.75 | 2695.34 | 2707.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:45:00 | 2734.90 | 2695.34 | 2707.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 2721.15 | 2695.59 | 2707.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:00:00 | 2721.15 | 2695.59 | 2707.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 2719.90 | 2695.83 | 2707.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-31 12:15:00 | 2710.00 | 2695.83 | 2707.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-31 15:15:00 | 2710.00 | 2696.14 | 2707.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-01 17:15:00 | 2738.00 | 2696.69 | 2708.12 | SL hit (close>static) qty=1.00 sl=2726.10 alert=retest2 |

### Cycle 4 — BUY (started 2025-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 10:15:00 | 2599.00 | 2261.76 | 2260.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 11:15:00 | 2651.20 | 2313.89 | 2288.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 2591.80 | 2684.31 | 2550.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 2591.80 | 2684.31 | 2550.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 2591.80 | 2684.31 | 2550.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 12:00:00 | 2675.60 | 2677.10 | 2552.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 11:15:00 | 2668.50 | 2678.66 | 2556.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 09:15:00 | 2747.40 | 2678.71 | 2559.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-16 14:15:00 | 2935.35 | 2708.94 | 2586.89 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 11:15:00 | 2671.30 | 3098.00 | 3099.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 10:15:00 | 2643.40 | 3072.66 | 3086.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 10:15:00 | 2859.20 | 2845.44 | 2920.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 11:00:00 | 2859.20 | 2845.44 | 2920.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 2628.00 | 2557.23 | 2623.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:00:00 | 2628.00 | 2557.23 | 2623.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 2621.00 | 2557.86 | 2623.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 2596.00 | 2557.86 | 2623.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 2591.00 | 2558.19 | 2623.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:15:00 | 2580.10 | 2558.19 | 2623.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:45:00 | 2576.00 | 2558.37 | 2623.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 11:45:00 | 2576.00 | 2558.55 | 2622.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 11:45:00 | 2580.40 | 2560.84 | 2621.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 10:15:00 | 2451.09 | 2542.55 | 2596.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 10:15:00 | 2447.20 | 2542.55 | 2596.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 10:15:00 | 2447.20 | 2542.55 | 2596.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 10:15:00 | 2451.38 | 2542.55 | 2596.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 2520.00 | 2509.26 | 2565.78 | SL hit (close>ema200) qty=0.50 sl=2509.26 alert=retest2 |

### Cycle 6 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 2604.50 | 2506.42 | 2506.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 2623.80 | 2509.34 | 2507.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 2532.50 | 2541.51 | 2525.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 2532.50 | 2541.51 | 2525.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 2532.50 | 2541.51 | 2525.50 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2026-03-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 11:15:00 | 2399.80 | 2514.88 | 2515.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 2390.10 | 2496.18 | 2505.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 09:15:00 | 2414.00 | 2400.01 | 2443.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-10 09:45:00 | 2422.60 | 2400.01 | 2443.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 2446.40 | 2400.37 | 2441.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 11:00:00 | 2446.40 | 2400.37 | 2441.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 11:15:00 | 2422.00 | 2400.59 | 2441.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 14:30:00 | 2416.60 | 2401.16 | 2441.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 10:15:00 | 2420.80 | 2401.45 | 2441.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 09:30:00 | 2408.50 | 2401.86 | 2439.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 15:15:00 | 2420.80 | 2402.32 | 2439.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 2455.60 | 2403.03 | 2439.15 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 2455.60 | 2403.03 | 2439.15 | SL hit (close>static) qty=1.00 sl=2453.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-08 13:15:00 | 2013.40 | 2024-05-16 12:15:00 | 2037.45 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-05-08 14:15:00 | 2014.05 | 2024-05-16 12:15:00 | 2037.45 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-05-09 10:00:00 | 2009.00 | 2024-05-16 12:15:00 | 2037.45 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-05-09 10:30:00 | 2015.00 | 2024-05-16 12:15:00 | 2037.45 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-10-04 10:30:00 | 2758.65 | 2024-10-07 09:15:00 | 2708.45 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2024-10-04 14:15:00 | 2760.40 | 2024-10-07 09:15:00 | 2708.45 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2024-10-09 09:15:00 | 2779.05 | 2024-10-14 12:15:00 | 2719.45 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2024-10-11 11:15:00 | 2757.50 | 2024-10-14 12:15:00 | 2719.45 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-10-15 09:30:00 | 2750.00 | 2024-10-15 12:15:00 | 2696.95 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-10-17 09:30:00 | 2742.50 | 2024-10-17 10:15:00 | 2691.20 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2024-10-31 12:15:00 | 2710.00 | 2024-11-01 17:15:00 | 2738.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-10-31 15:15:00 | 2710.00 | 2024-11-01 17:15:00 | 2738.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-11-04 09:15:00 | 2693.65 | 2024-11-08 10:15:00 | 2558.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-04 09:15:00 | 2693.65 | 2024-11-18 09:15:00 | 2424.29 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-04-08 12:00:00 | 2675.60 | 2025-04-16 14:15:00 | 2935.35 | TARGET_HIT | 1.00 | 9.71% |
| BUY | retest2 | 2025-04-09 11:15:00 | 2668.50 | 2025-04-17 09:15:00 | 2943.16 | TARGET_HIT | 1.00 | 10.29% |
| BUY | retest2 | 2025-04-11 09:15:00 | 2747.40 | 2025-04-24 09:15:00 | 3022.14 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-12-12 10:15:00 | 2580.10 | 2025-12-29 10:15:00 | 2451.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 10:45:00 | 2576.00 | 2025-12-29 10:15:00 | 2447.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 11:45:00 | 2576.00 | 2025-12-29 10:15:00 | 2447.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-15 11:45:00 | 2580.40 | 2025-12-29 10:15:00 | 2451.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 10:15:00 | 2580.10 | 2026-01-07 09:15:00 | 2520.00 | STOP_HIT | 0.50 | 2.33% |
| SELL | retest2 | 2025-12-12 10:45:00 | 2576.00 | 2026-01-07 09:15:00 | 2520.00 | STOP_HIT | 0.50 | 2.17% |
| SELL | retest2 | 2025-12-12 11:45:00 | 2576.00 | 2026-01-07 09:15:00 | 2520.00 | STOP_HIT | 0.50 | 2.17% |
| SELL | retest2 | 2025-12-15 11:45:00 | 2580.40 | 2026-01-07 09:15:00 | 2520.00 | STOP_HIT | 0.50 | 2.34% |
| SELL | retest2 | 2026-02-03 12:45:00 | 2413.90 | 2026-02-06 14:15:00 | 2483.60 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2026-02-04 09:15:00 | 2403.50 | 2026-02-06 14:15:00 | 2483.60 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2026-02-05 15:15:00 | 2425.10 | 2026-02-06 14:15:00 | 2483.60 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2026-04-13 14:30:00 | 2416.60 | 2026-04-17 09:15:00 | 2455.60 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-04-15 10:15:00 | 2420.80 | 2026-04-17 09:15:00 | 2455.60 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2026-04-16 09:30:00 | 2408.50 | 2026-04-17 09:15:00 | 2455.60 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2026-04-16 15:15:00 | 2420.80 | 2026-04-17 09:15:00 | 2455.60 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2026-04-20 09:15:00 | 2436.70 | 2026-04-22 13:15:00 | 2468.90 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-04-20 12:00:00 | 2429.60 | 2026-04-22 13:15:00 | 2468.90 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-04-21 09:45:00 | 2436.90 | 2026-04-22 13:15:00 | 2468.90 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-04-21 10:30:00 | 2436.80 | 2026-04-22 13:15:00 | 2468.90 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-04-28 12:00:00 | 2432.00 | 2026-05-07 09:15:00 | 2451.70 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2026-05-06 12:15:00 | 2431.00 | 2026-05-07 09:15:00 | 2451.70 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-05-06 13:15:00 | 2421.00 | 2026-05-07 09:15:00 | 2451.70 | STOP_HIT | 1.00 | -1.27% |
