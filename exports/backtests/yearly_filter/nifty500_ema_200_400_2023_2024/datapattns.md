# Data Patterns (India) Ltd. (DATAPATTNS)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 4118.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 2 |
| ALERT3 | 50 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 58 |
| PARTIAL | 17 |
| TARGET_HIT | 19 |
| STOP_HIT | 40 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 75 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 39 / 36
- **Target hits / Stop hits / Partials:** 18 / 40 / 17
- **Avg / median % per leg:** 1.96% / 1.41%
- **Sum % (uncompounded):** 146.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 12 | 75.0% | 12 | 4 | 0 | 6.47% | 103.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 12 | 75.0% | 12 | 4 | 0 | 6.47% | 103.6% |
| SELL (all) | 59 | 27 | 45.8% | 6 | 36 | 17 | 0.73% | 43.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 59 | 27 | 45.8% | 6 | 36 | 17 | 0.73% | 43.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 75 | 39 | 52.0% | 18 | 40 | 17 | 1.96% | 146.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 13:15:00 | 1848.45 | 2078.74 | 2079.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-26 09:15:00 | 1758.15 | 2070.68 | 2074.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-24 09:15:00 | 1937.75 | 1921.81 | 1975.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-24 10:00:00 | 1937.75 | 1921.81 | 1975.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 14:15:00 | 1985.40 | 1923.76 | 1974.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 15:00:00 | 1985.40 | 1923.76 | 1974.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 15:15:00 | 1989.00 | 1924.41 | 1974.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 09:15:00 | 1981.95 | 1924.41 | 1974.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 09:15:00 | 1981.70 | 1926.48 | 1973.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 09:45:00 | 1992.05 | 1926.48 | 1973.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 10:15:00 | 1977.60 | 1926.99 | 1973.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-29 15:00:00 | 1966.20 | 1928.85 | 1973.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-30 09:15:00 | 1989.00 | 1929.84 | 1973.91 | SL hit (close>static) qty=1.00 sl=1986.30 alert=retest2 |

### Cycle 2 — BUY (started 2024-02-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 13:15:00 | 2179.00 | 1947.34 | 1946.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-22 14:15:00 | 2235.00 | 1950.20 | 1947.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 09:15:00 | 2260.00 | 2328.45 | 2177.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-13 10:00:00 | 2260.00 | 2328.45 | 2177.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 09:15:00 | 2281.10 | 2321.01 | 2179.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-14 11:00:00 | 2344.90 | 2321.25 | 2179.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-14 12:15:00 | 2339.40 | 2321.24 | 2180.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-14 13:00:00 | 2336.00 | 2321.38 | 2181.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-14 14:30:00 | 2338.95 | 2322.28 | 2183.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-01 12:15:00 | 2579.39 | 2342.79 | 2233.73 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 09:15:00 | 2879.00 | 2993.46 | 2993.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 15:15:00 | 2865.00 | 2986.37 | 2990.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 15:15:00 | 2533.00 | 2531.42 | 2662.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-17 09:15:00 | 2482.50 | 2531.42 | 2662.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 2531.65 | 2338.80 | 2442.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:45:00 | 2534.45 | 2338.80 | 2442.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 2497.00 | 2340.38 | 2443.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 12:15:00 | 2480.95 | 2341.89 | 2443.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 09:30:00 | 2479.00 | 2347.92 | 2443.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 10:00:00 | 2461.75 | 2347.92 | 2443.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 14:00:00 | 2472.70 | 2352.79 | 2444.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 09:15:00 | 2537.60 | 2357.54 | 2445.51 | SL hit (close>static) qty=1.00 sl=2532.00 alert=retest2 |

### Cycle 4 — BUY (started 2024-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 11:15:00 | 2553.30 | 2502.64 | 2502.56 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 13:15:00 | 2462.70 | 2502.85 | 2502.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 14:15:00 | 2418.60 | 2502.02 | 2502.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-01 13:15:00 | 2500.45 | 2497.30 | 2499.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-01 13:15:00 | 2500.45 | 2497.30 | 2499.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 13:15:00 | 2500.45 | 2497.30 | 2499.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 14:00:00 | 2500.45 | 2497.30 | 2499.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 14:15:00 | 2499.75 | 2497.32 | 2499.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 15:15:00 | 2499.00 | 2497.32 | 2499.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 15:15:00 | 2499.00 | 2497.34 | 2499.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 09:15:00 | 2501.05 | 2497.34 | 2499.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 2498.00 | 2497.35 | 2499.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-02 11:15:00 | 2482.10 | 2497.27 | 2499.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-02 12:30:00 | 2485.45 | 2497.04 | 2499.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-02 13:00:00 | 2484.85 | 2497.04 | 2499.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-03 09:15:00 | 2524.00 | 2497.07 | 2499.70 | SL hit (close>static) qty=1.00 sl=2513.90 alert=retest2 |

### Cycle 6 — BUY (started 2025-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 10:15:00 | 2345.00 | 1864.70 | 1864.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 11:15:00 | 2407.40 | 1870.10 | 1867.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 09:15:00 | 2807.20 | 2850.06 | 2607.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 10:00:00 | 2807.20 | 2850.06 | 2607.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 2755.00 | 2888.10 | 2746.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 2637.10 | 2888.10 | 2746.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 2674.60 | 2885.98 | 2745.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 09:15:00 | 2762.20 | 2875.77 | 2744.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 11:15:00 | 2610.70 | 2841.84 | 2745.68 | SL hit (close<static) qty=1.00 sl=2627.10 alert=retest2 |

### Cycle 7 — SELL (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 10:15:00 | 2512.00 | 2685.04 | 2685.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 11:15:00 | 2505.70 | 2683.25 | 2684.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 2574.70 | 2568.39 | 2610.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 10:00:00 | 2574.70 | 2568.39 | 2610.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 2614.20 | 2568.85 | 2609.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 13:45:00 | 2566.10 | 2569.63 | 2608.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 09:30:00 | 2570.00 | 2570.12 | 2608.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 11:15:00 | 2712.60 | 2572.07 | 2608.87 | SL hit (close>static) qty=1.00 sl=2689.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 12:15:00 | 2836.00 | 2638.84 | 2638.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 13:15:00 | 2843.90 | 2640.88 | 2639.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 11:15:00 | 2661.00 | 2678.84 | 2660.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 11:15:00 | 2661.00 | 2678.84 | 2660.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 2661.00 | 2678.84 | 2660.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:30:00 | 2661.40 | 2678.84 | 2660.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 12:15:00 | 2657.00 | 2678.62 | 2660.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 13:00:00 | 2657.00 | 2678.62 | 2660.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 2642.40 | 2678.26 | 2660.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 13:30:00 | 2641.20 | 2678.26 | 2660.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 2660.00 | 2677.62 | 2660.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:15:00 | 2640.00 | 2677.62 | 2660.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 2613.10 | 2676.98 | 2660.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:45:00 | 2616.10 | 2676.98 | 2660.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 2709.70 | 2713.73 | 2686.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 13:15:00 | 2727.40 | 2713.71 | 2687.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 2739.00 | 2743.86 | 2711.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 13:00:00 | 2741.70 | 2743.31 | 2712.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 2636.10 | 2740.07 | 2713.35 | SL hit (close<static) qty=1.00 sl=2685.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 2570.30 | 2779.18 | 2779.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 2532.00 | 2770.81 | 2775.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 11:15:00 | 2720.60 | 2711.70 | 2741.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-24 11:45:00 | 2730.10 | 2711.70 | 2741.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 2716.10 | 2710.70 | 2740.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 12:30:00 | 2708.10 | 2710.70 | 2740.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 12:15:00 | 2572.69 | 2702.46 | 2733.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 2705.50 | 2682.94 | 2719.71 | SL hit (close>ema200) qty=0.50 sl=2682.94 alert=retest2 |

### Cycle 10 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 2821.90 | 2655.62 | 2655.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 10:15:00 | 2863.10 | 2662.65 | 2658.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 13:15:00 | 3075.50 | 3076.66 | 2920.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:00:00 | 3075.50 | 3076.66 | 2920.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 2958.30 | 3127.96 | 3000.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 09:30:00 | 2966.80 | 3127.96 | 3000.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 2909.00 | 3125.78 | 2999.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:30:00 | 2912.10 | 3125.78 | 2999.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 3069.10 | 3120.95 | 2999.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 14:45:00 | 3008.00 | 3120.95 | 2999.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 3089.90 | 3119.97 | 3000.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:30:00 | 3119.60 | 3119.58 | 3001.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 11:30:00 | 3123.60 | 3119.81 | 3001.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 13:45:00 | 3122.80 | 3120.05 | 3003.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-10 09:15:00 | 3431.56 | 3155.53 | 3034.78 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-11-29 15:00:00 | 1966.20 | 2023-11-30 09:15:00 | 1989.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2023-12-20 13:15:00 | 1967.00 | 2023-12-29 12:15:00 | 1871.26 | PARTIAL | 0.50 | 4.87% |
| SELL | retest2 | 2023-12-22 10:30:00 | 1969.75 | 2023-12-29 14:15:00 | 1868.65 | PARTIAL | 0.50 | 5.13% |
| SELL | retest2 | 2023-12-26 09:30:00 | 1967.45 | 2023-12-29 14:15:00 | 1869.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-20 13:15:00 | 1967.00 | 2024-01-08 10:15:00 | 1939.30 | STOP_HIT | 0.50 | 1.41% |
| SELL | retest2 | 2023-12-22 10:30:00 | 1969.75 | 2024-01-08 10:15:00 | 1939.30 | STOP_HIT | 0.50 | 1.55% |
| SELL | retest2 | 2023-12-26 09:30:00 | 1967.45 | 2024-01-08 10:15:00 | 1939.30 | STOP_HIT | 0.50 | 1.43% |
| SELL | retest2 | 2024-01-09 14:15:00 | 1950.30 | 2024-01-12 11:15:00 | 1990.60 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2024-01-09 15:00:00 | 1960.90 | 2024-01-12 11:15:00 | 1990.60 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2024-01-10 10:15:00 | 1950.95 | 2024-01-12 11:15:00 | 1990.60 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2024-01-11 11:30:00 | 1960.75 | 2024-01-12 11:15:00 | 1990.60 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-01-16 10:15:00 | 1954.85 | 2024-01-17 13:15:00 | 1977.45 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-01-16 12:00:00 | 1953.60 | 2024-01-17 13:15:00 | 1977.45 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-01-16 12:30:00 | 1952.00 | 2024-01-17 13:15:00 | 1977.45 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-01-18 09:30:00 | 1925.05 | 2024-01-23 14:15:00 | 1834.45 | PARTIAL | 0.50 | 4.71% |
| SELL | retest2 | 2024-01-19 11:00:00 | 1931.00 | 2024-01-24 09:15:00 | 1828.80 | PARTIAL | 0.50 | 5.29% |
| SELL | retest2 | 2024-01-18 09:30:00 | 1925.05 | 2024-01-30 09:15:00 | 1937.85 | STOP_HIT | 0.50 | -0.66% |
| SELL | retest2 | 2024-01-19 11:00:00 | 1931.00 | 2024-01-30 09:15:00 | 1937.85 | STOP_HIT | 0.50 | -0.35% |
| SELL | retest2 | 2024-01-31 09:30:00 | 1928.65 | 2024-02-01 09:15:00 | 1984.00 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2024-02-01 12:00:00 | 1932.05 | 2024-02-12 09:15:00 | 1841.72 | PARTIAL | 0.50 | 4.68% |
| SELL | retest2 | 2024-02-01 13:15:00 | 1938.65 | 2024-02-13 09:15:00 | 1835.45 | PARTIAL | 0.50 | 5.32% |
| SELL | retest2 | 2024-02-05 09:30:00 | 1887.45 | 2024-02-13 09:15:00 | 1793.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-07 09:30:00 | 1902.25 | 2024-02-13 09:15:00 | 1807.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-07 10:15:00 | 1902.10 | 2024-02-13 09:15:00 | 1806.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-08 09:30:00 | 1897.00 | 2024-02-13 09:15:00 | 1802.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-01 12:00:00 | 1932.05 | 2024-02-16 09:15:00 | 2001.50 | STOP_HIT | 0.50 | -3.59% |
| SELL | retest2 | 2024-02-01 13:15:00 | 1938.65 | 2024-02-16 09:15:00 | 2001.50 | STOP_HIT | 0.50 | -3.24% |
| SELL | retest2 | 2024-02-05 09:30:00 | 1887.45 | 2024-02-16 09:15:00 | 2001.50 | STOP_HIT | 0.50 | -6.04% |
| SELL | retest2 | 2024-02-07 09:30:00 | 1902.25 | 2024-02-16 09:15:00 | 2001.50 | STOP_HIT | 0.50 | -5.22% |
| SELL | retest2 | 2024-02-07 10:15:00 | 1902.10 | 2024-02-16 09:15:00 | 2001.50 | STOP_HIT | 0.50 | -5.23% |
| SELL | retest2 | 2024-02-08 09:30:00 | 1897.00 | 2024-02-16 09:15:00 | 2001.50 | STOP_HIT | 0.50 | -5.51% |
| BUY | retest2 | 2024-03-14 11:00:00 | 2344.90 | 2024-04-01 12:15:00 | 2579.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-14 12:15:00 | 2339.40 | 2024-04-01 12:15:00 | 2573.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-14 13:00:00 | 2336.00 | 2024-04-01 12:15:00 | 2569.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-14 14:30:00 | 2338.95 | 2024-04-01 12:15:00 | 2572.84 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-18 09:15:00 | 3027.45 | 2024-07-05 09:15:00 | 3281.14 | TARGET_HIT | 1.00 | 8.38% |
| BUY | retest2 | 2024-06-19 09:30:00 | 3004.00 | 2024-07-05 09:15:00 | 3288.45 | TARGET_HIT | 1.00 | 9.47% |
| BUY | retest2 | 2024-06-21 09:15:00 | 2982.85 | 2024-07-05 10:15:00 | 3330.20 | TARGET_HIT | 1.00 | 11.64% |
| BUY | retest2 | 2024-06-24 10:00:00 | 2989.50 | 2024-07-05 10:15:00 | 3304.40 | TARGET_HIT | 1.00 | 10.53% |
| SELL | retest2 | 2024-11-28 12:15:00 | 2480.95 | 2024-12-02 09:15:00 | 2537.60 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-11-29 09:30:00 | 2479.00 | 2024-12-02 09:15:00 | 2537.60 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2024-11-29 10:00:00 | 2461.75 | 2024-12-02 09:15:00 | 2537.60 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2024-11-29 14:00:00 | 2472.70 | 2024-12-02 09:15:00 | 2537.60 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-01-02 11:15:00 | 2482.10 | 2025-01-03 09:15:00 | 2524.00 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-01-02 12:30:00 | 2485.45 | 2025-01-03 09:15:00 | 2524.00 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-01-02 13:00:00 | 2484.85 | 2025-01-03 09:15:00 | 2524.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-01-03 14:45:00 | 2481.50 | 2025-01-06 14:15:00 | 2357.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-06 10:30:00 | 2427.00 | 2025-01-08 12:15:00 | 2305.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-03 14:45:00 | 2481.50 | 2025-01-10 09:15:00 | 2233.35 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-06 10:30:00 | 2427.00 | 2025-01-13 09:15:00 | 2184.30 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-07-22 09:15:00 | 2762.20 | 2025-07-28 11:15:00 | 2610.70 | STOP_HIT | 1.00 | -5.48% |
| SELL | retest2 | 2025-09-11 13:45:00 | 2566.10 | 2025-09-12 11:15:00 | 2712.60 | STOP_HIT | 1.00 | -5.71% |
| SELL | retest2 | 2025-09-12 09:30:00 | 2570.00 | 2025-09-12 11:15:00 | 2712.60 | STOP_HIT | 1.00 | -5.55% |
| BUY | retest2 | 2025-10-16 13:15:00 | 2727.40 | 2025-11-06 09:15:00 | 2636.10 | STOP_HIT | 1.00 | -3.35% |
| BUY | retest2 | 2025-10-31 09:15:00 | 2739.00 | 2025-11-06 09:15:00 | 2636.10 | STOP_HIT | 1.00 | -3.76% |
| BUY | retest2 | 2025-10-31 13:00:00 | 2741.70 | 2025-11-06 09:15:00 | 2636.10 | STOP_HIT | 1.00 | -3.85% |
| BUY | retest2 | 2025-11-11 09:15:00 | 2746.80 | 2025-11-13 09:15:00 | 3021.48 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-12-26 12:30:00 | 2708.10 | 2025-12-30 12:15:00 | 2572.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-26 12:30:00 | 2708.10 | 2026-01-05 09:15:00 | 2705.50 | STOP_HIT | 0.50 | 0.10% |
| SELL | retest2 | 2026-01-06 09:15:00 | 2707.90 | 2026-01-12 11:15:00 | 2572.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 14:00:00 | 2711.00 | 2026-01-12 11:15:00 | 2575.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 15:00:00 | 2683.00 | 2026-01-16 09:15:00 | 2548.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 09:15:00 | 2707.90 | 2026-01-20 09:15:00 | 2437.11 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-08 14:00:00 | 2711.00 | 2026-01-20 09:15:00 | 2439.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-08 15:00:00 | 2683.00 | 2026-01-20 09:15:00 | 2414.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-29 09:30:00 | 2573.00 | 2026-01-30 09:15:00 | 2675.20 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2026-01-29 11:45:00 | 2576.60 | 2026-01-30 09:15:00 | 2675.20 | STOP_HIT | 1.00 | -3.83% |
| SELL | retest2 | 2026-01-29 14:00:00 | 2571.10 | 2026-01-30 09:15:00 | 2675.20 | STOP_HIT | 1.00 | -4.05% |
| SELL | retest2 | 2026-02-01 11:45:00 | 2580.00 | 2026-02-01 12:15:00 | 2322.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-04 09:45:00 | 2536.20 | 2026-02-06 10:15:00 | 2734.40 | STOP_HIT | 1.00 | -7.81% |
| SELL | retest2 | 2026-02-04 10:15:00 | 2535.60 | 2026-02-06 10:15:00 | 2734.40 | STOP_HIT | 1.00 | -7.84% |
| SELL | retest2 | 2026-02-05 09:15:00 | 2537.80 | 2026-02-06 10:15:00 | 2734.40 | STOP_HIT | 1.00 | -7.75% |
| BUY | retest2 | 2026-04-06 10:30:00 | 3119.60 | 2026-04-10 09:15:00 | 3431.56 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 11:30:00 | 3123.60 | 2026-04-10 09:15:00 | 3435.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 13:45:00 | 3122.80 | 2026-04-10 09:15:00 | 3435.08 | TARGET_HIT | 1.00 | 10.00% |
