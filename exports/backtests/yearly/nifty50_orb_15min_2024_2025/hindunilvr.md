# HINDUNILVR (HINDUNILVR)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (35368 bars)
- **Last close:** 2286.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 89 |
| ENTRY2 | 0 |
| PARTIAL | 36 |
| TARGET_HIT | 17 |
| STOP_HIT | 72 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 125 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 53 / 72
- **Target hits / Stop hits / Partials:** 17 / 72 / 36
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 15.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 68 | 34 | 50.0% | 12 | 34 | 22 | 0.21% | 14.0% |
| BUY @ 2nd Alert (retest1) | 68 | 34 | 50.0% | 12 | 34 | 22 | 0.21% | 14.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 57 | 19 | 33.3% | 5 | 38 | 14 | 0.03% | 1.6% |
| SELL @ 2nd Alert (retest1) | 57 | 19 | 33.3% | 5 | 38 | 14 | 0.03% | 1.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 125 | 53 | 42.4% | 17 | 72 | 36 | 0.12% | 15.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-15 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 10:30:00 | 2291.95 | 2302.46 | 0.00 | ORB-short ORB[2303.95,2317.33] vol=1.9x ATR=3.53 |
| Stop hit — per-position SL triggered | 2024-05-15 11:00:00 | 2295.48 | 2299.87 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 09:30:00 | 2277.79 | 2284.83 | 0.00 | ORB-short ORB[2282.12,2298.84] vol=2.0x ATR=4.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 10:00:00 | 2270.85 | 2279.17 | 0.00 | T1 1.5R @ 2270.85 |
| Target hit | 2024-05-16 11:35:00 | 2274.30 | 2274.26 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — BUY (started 2024-05-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-22 09:40:00 | 2306.76 | 2296.74 | 0.00 | ORB-long ORB[2272.97,2292.94] vol=1.6x ATR=4.18 |
| Stop hit — per-position SL triggered | 2024-05-22 09:45:00 | 2302.58 | 2297.52 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-27 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 10:25:00 | 2355.15 | 2338.37 | 0.00 | ORB-long ORB[2325.30,2339.37] vol=2.0x ATR=5.91 |
| Stop hit — per-position SL triggered | 2024-05-27 11:00:00 | 2349.24 | 2341.86 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 09:30:00 | 2341.28 | 2346.50 | 0.00 | ORB-short ORB[2342.12,2356.88] vol=2.9x ATR=5.29 |
| Stop hit — per-position SL triggered | 2024-05-28 09:45:00 | 2346.57 | 2346.21 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 10:40:00 | 2319.45 | 2324.17 | 0.00 | ORB-short ORB[2321.61,2333.71] vol=1.6x ATR=4.87 |
| Stop hit — per-position SL triggered | 2024-05-30 12:00:00 | 2324.32 | 2322.54 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-04 09:40:00 | 2350.48 | 2326.20 | 0.00 | ORB-long ORB[2303.86,2335.43] vol=3.2x ATR=10.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:45:00 | 2365.58 | 2335.01 | 0.00 | T1 1.5R @ 2365.58 |
| Target hit | 2024-06-04 15:20:00 | 2456.62 | 2412.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2024-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 11:15:00 | 2487.95 | 2499.33 | 0.00 | ORB-short ORB[2490.70,2526.07] vol=2.0x ATR=5.34 |
| Stop hit — per-position SL triggered | 2024-06-12 11:20:00 | 2493.29 | 2499.11 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 10:50:00 | 2437.54 | 2440.79 | 0.00 | ORB-short ORB[2439.75,2454.26] vol=2.4x ATR=4.66 |
| Stop hit — per-position SL triggered | 2024-06-19 10:55:00 | 2442.20 | 2440.68 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 10:45:00 | 2403.50 | 2415.43 | 0.00 | ORB-short ORB[2416.39,2437.54] vol=1.9x ATR=4.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-21 11:30:00 | 2396.19 | 2410.86 | 0.00 | T1 1.5R @ 2396.19 |
| Stop hit — per-position SL triggered | 2024-06-21 13:20:00 | 2403.50 | 2403.77 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:15:00 | 2413.63 | 2400.77 | 0.00 | ORB-long ORB[2395.24,2406.06] vol=2.0x ATR=4.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 10:25:00 | 2419.90 | 2405.08 | 0.00 | T1 1.5R @ 2419.90 |
| Stop hit — per-position SL triggered | 2024-06-26 10:50:00 | 2413.63 | 2409.81 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-06-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 10:05:00 | 2425.14 | 2410.62 | 0.00 | ORB-long ORB[2398.39,2414.91] vol=1.9x ATR=5.49 |
| Stop hit — per-position SL triggered | 2024-06-27 10:15:00 | 2419.65 | 2412.13 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:35:00 | 2452.39 | 2442.23 | 0.00 | ORB-long ORB[2410.09,2432.62] vol=5.7x ATR=7.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 09:50:00 | 2462.96 | 2450.44 | 0.00 | T1 1.5R @ 2462.96 |
| Target hit | 2024-07-01 15:20:00 | 2459.72 | 2461.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2024-07-02 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 10:30:00 | 2449.34 | 2455.06 | 0.00 | ORB-short ORB[2455.10,2473.93] vol=3.4x ATR=5.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 12:45:00 | 2441.45 | 2451.81 | 0.00 | T1 1.5R @ 2441.45 |
| Stop hit — per-position SL triggered | 2024-07-02 12:50:00 | 2449.34 | 2451.61 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-04 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 10:40:00 | 2483.47 | 2475.53 | 0.00 | ORB-long ORB[2456.23,2475.56] vol=2.4x ATR=6.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 10:50:00 | 2492.58 | 2477.44 | 0.00 | T1 1.5R @ 2492.58 |
| Stop hit — per-position SL triggered | 2024-07-04 11:05:00 | 2483.47 | 2480.18 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 11:10:00 | 2479.34 | 2471.47 | 0.00 | ORB-long ORB[2455.19,2478.75] vol=3.2x ATR=3.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 12:05:00 | 2485.25 | 2475.09 | 0.00 | T1 1.5R @ 2485.25 |
| Target hit | 2024-07-05 15:20:00 | 2506.20 | 2488.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2024-07-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 10:50:00 | 2546.48 | 2527.68 | 0.00 | ORB-long ORB[2498.52,2535.66] vol=1.7x ATR=8.17 |
| Stop hit — per-position SL triggered | 2024-07-08 10:55:00 | 2538.31 | 2528.05 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:40:00 | 2538.12 | 2553.19 | 0.00 | ORB-short ORB[2538.86,2560.00] vol=2.2x ATR=7.13 |
| Stop hit — per-position SL triggered | 2024-07-10 11:50:00 | 2545.25 | 2548.11 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 10:50:00 | 2577.22 | 2570.92 | 0.00 | ORB-long ORB[2558.53,2573.28] vol=2.6x ATR=5.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 10:55:00 | 2584.89 | 2572.43 | 0.00 | T1 1.5R @ 2584.89 |
| Stop hit — per-position SL triggered | 2024-07-12 11:15:00 | 2577.22 | 2575.09 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 11:05:00 | 2613.66 | 2600.94 | 0.00 | ORB-long ORB[2581.69,2605.74] vol=1.6x ATR=6.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 11:10:00 | 2623.16 | 2603.81 | 0.00 | T1 1.5R @ 2623.16 |
| Target hit | 2024-07-16 15:20:00 | 2642.98 | 2639.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — BUY (started 2024-07-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 11:10:00 | 2717.49 | 2701.93 | 0.00 | ORB-long ORB[2692.41,2711.93] vol=2.5x ATR=5.97 |
| Stop hit — per-position SL triggered | 2024-07-23 11:15:00 | 2711.52 | 2703.53 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-07-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-25 10:50:00 | 2648.04 | 2667.05 | 0.00 | ORB-short ORB[2660.98,2680.36] vol=1.8x ATR=6.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 11:10:00 | 2637.79 | 2661.63 | 0.00 | T1 1.5R @ 2637.79 |
| Stop hit — per-position SL triggered | 2024-07-25 12:40:00 | 2648.04 | 2652.81 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-07-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-26 10:55:00 | 2646.03 | 2656.25 | 0.00 | ORB-short ORB[2646.12,2679.52] vol=2.0x ATR=5.73 |
| Stop hit — per-position SL triggered | 2024-07-26 11:00:00 | 2651.76 | 2656.08 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-07-31 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-31 10:45:00 | 2637.62 | 2646.43 | 0.00 | ORB-short ORB[2649.91,2665.40] vol=1.8x ATR=5.13 |
| Stop hit — per-position SL triggered | 2024-07-31 11:35:00 | 2642.75 | 2643.01 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-01 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 10:25:00 | 2661.08 | 2666.28 | 0.00 | ORB-short ORB[2668.45,2678.19] vol=2.0x ATR=5.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 11:10:00 | 2652.79 | 2663.86 | 0.00 | T1 1.5R @ 2652.79 |
| Stop hit — per-position SL triggered | 2024-08-01 12:30:00 | 2661.08 | 2662.58 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 10:10:00 | 2726.49 | 2724.24 | 0.00 | ORB-long ORB[2698.16,2721.28] vol=5.0x ATR=4.94 |
| Stop hit — per-position SL triggered | 2024-08-21 10:35:00 | 2721.55 | 2724.27 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-08-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-22 10:45:00 | 2742.47 | 2750.79 | 0.00 | ORB-short ORB[2747.64,2761.66] vol=1.8x ATR=5.49 |
| Stop hit — per-position SL triggered | 2024-08-22 13:15:00 | 2747.96 | 2746.08 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 10:50:00 | 2786.15 | 2773.19 | 0.00 | ORB-long ORB[2757.82,2773.76] vol=1.8x ATR=5.01 |
| Stop hit — per-position SL triggered | 2024-08-26 11:05:00 | 2781.14 | 2776.14 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-08-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 10:50:00 | 2710.01 | 2713.98 | 0.00 | ORB-short ORB[2710.70,2723.83] vol=1.8x ATR=4.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-28 10:55:00 | 2703.92 | 2711.87 | 0.00 | T1 1.5R @ 2703.92 |
| Stop hit — per-position SL triggered | 2024-08-28 11:45:00 | 2710.01 | 2708.66 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-08-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 10:45:00 | 2738.15 | 2730.08 | 0.00 | ORB-long ORB[2700.18,2736.77] vol=3.1x ATR=4.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 11:15:00 | 2745.56 | 2731.94 | 0.00 | T1 1.5R @ 2745.56 |
| Stop hit — per-position SL triggered | 2024-08-29 11:25:00 | 2738.15 | 2732.32 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-09-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-02 10:50:00 | 2755.02 | 2752.98 | 0.00 | ORB-long ORB[2733.62,2748.38] vol=2.5x ATR=5.72 |
| Stop hit — per-position SL triggered | 2024-09-02 11:55:00 | 2749.30 | 2753.67 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:30:00 | 2757.23 | 2746.23 | 0.00 | ORB-long ORB[2729.69,2752.26] vol=1.6x ATR=5.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-03 09:40:00 | 2765.99 | 2750.66 | 0.00 | T1 1.5R @ 2765.99 |
| Stop hit — per-position SL triggered | 2024-09-03 09:45:00 | 2757.23 | 2752.54 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 10:05:00 | 2767.51 | 2753.25 | 0.00 | ORB-long ORB[2726.39,2752.21] vol=1.7x ATR=6.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 10:55:00 | 2776.80 | 2758.14 | 0.00 | T1 1.5R @ 2776.80 |
| Target hit | 2024-09-04 15:20:00 | 2794.81 | 2779.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — BUY (started 2024-09-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 09:30:00 | 2801.10 | 2789.47 | 0.00 | ORB-long ORB[2777.89,2797.27] vol=1.8x ATR=5.71 |
| Stop hit — per-position SL triggered | 2024-09-05 09:55:00 | 2795.39 | 2795.26 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-09-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 10:50:00 | 2886.34 | 2874.32 | 0.00 | ORB-long ORB[2852.65,2876.25] vol=2.9x ATR=5.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 11:00:00 | 2894.21 | 2876.86 | 0.00 | T1 1.5R @ 2894.21 |
| Stop hit — per-position SL triggered | 2024-09-11 11:35:00 | 2886.34 | 2884.94 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-09-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 10:25:00 | 2870.70 | 2865.99 | 0.00 | ORB-long ORB[2846.94,2869.27] vol=1.6x ATR=6.37 |
| Stop hit — per-position SL triggered | 2024-09-12 10:30:00 | 2864.33 | 2866.00 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-09-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 09:35:00 | 2844.78 | 2838.92 | 0.00 | ORB-long ORB[2826.43,2841.83] vol=2.2x ATR=5.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 09:55:00 | 2853.00 | 2841.99 | 0.00 | T1 1.5R @ 2853.00 |
| Target hit | 2024-09-19 12:35:00 | 2853.97 | 2863.35 | 0.00 | Trail-exit close<VWAP |

### Cycle 38 — BUY (started 2024-10-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-07 11:00:00 | 2806.91 | 2795.36 | 0.00 | ORB-long ORB[2781.97,2806.81] vol=2.0x ATR=8.76 |
| Stop hit — per-position SL triggered | 2024-10-07 11:15:00 | 2798.15 | 2796.25 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-10-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-09 10:50:00 | 2745.97 | 2749.22 | 0.00 | ORB-short ORB[2750.74,2779.71] vol=2.7x ATR=9.56 |
| Stop hit — per-position SL triggered | 2024-10-09 11:10:00 | 2755.53 | 2749.58 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-10-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-10 11:10:00 | 2700.62 | 2717.18 | 0.00 | ORB-short ORB[2708.44,2732.44] vol=3.8x ATR=7.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-10 12:05:00 | 2689.34 | 2706.05 | 0.00 | T1 1.5R @ 2689.34 |
| Stop hit — per-position SL triggered | 2024-10-10 13:10:00 | 2700.62 | 2701.53 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-10-15 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-15 10:25:00 | 2724.13 | 2733.52 | 0.00 | ORB-short ORB[2739.18,2753.59] vol=2.5x ATR=6.18 |
| Stop hit — per-position SL triggered | 2024-10-15 10:50:00 | 2730.31 | 2732.38 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-10-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 11:05:00 | 2712.96 | 2724.82 | 0.00 | ORB-short ORB[2719.85,2736.57] vol=1.9x ATR=6.60 |
| Stop hit — per-position SL triggered | 2024-10-17 11:35:00 | 2719.56 | 2722.38 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-10-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 10:45:00 | 2507.18 | 2514.66 | 0.00 | ORB-short ORB[2508.41,2543.48] vol=3.4x ATR=6.26 |
| Stop hit — per-position SL triggered | 2024-10-29 10:50:00 | 2513.44 | 2514.21 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-11-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-04 10:00:00 | 2484.75 | 2496.08 | 0.00 | ORB-short ORB[2487.26,2505.41] vol=1.6x ATR=6.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-04 10:10:00 | 2475.40 | 2493.50 | 0.00 | T1 1.5R @ 2475.40 |
| Stop hit — per-position SL triggered | 2024-11-04 12:00:00 | 2484.75 | 2480.84 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-11-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-08 10:30:00 | 2449.59 | 2439.44 | 0.00 | ORB-long ORB[2427.01,2446.24] vol=4.7x ATR=5.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-08 10:55:00 | 2457.51 | 2442.01 | 0.00 | T1 1.5R @ 2457.51 |
| Target hit | 2024-11-08 13:35:00 | 2449.93 | 2450.60 | 0.00 | Trail-exit close<VWAP |

### Cycle 46 — SELL (started 2024-11-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 10:50:00 | 2436.95 | 2439.21 | 0.00 | ORB-short ORB[2438.27,2456.18] vol=2.1x ATR=5.28 |
| Stop hit — per-position SL triggered | 2024-11-12 10:55:00 | 2442.23 | 2439.38 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-11-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-14 11:05:00 | 2388.40 | 2395.01 | 0.00 | ORB-short ORB[2395.44,2424.45] vol=1.9x ATR=6.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 11:50:00 | 2378.24 | 2393.11 | 0.00 | T1 1.5R @ 2378.24 |
| Target hit | 2024-11-14 15:20:00 | 2347.53 | 2370.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 2412.11 | 2424.49 | 0.00 | ORB-short ORB[2419.83,2435.57] vol=3.1x ATR=5.30 |
| Stop hit — per-position SL triggered | 2024-12-05 11:05:00 | 2417.41 | 2423.98 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-12-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 10:45:00 | 2440.49 | 2445.03 | 0.00 | ORB-short ORB[2443.49,2455.05] vol=2.1x ATR=5.66 |
| Stop hit — per-position SL triggered | 2024-12-06 15:20:00 | 2442.31 | 2442.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — SELL (started 2024-12-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 10:20:00 | 2329.78 | 2349.48 | 0.00 | ORB-short ORB[2346.10,2360.76] vol=1.8x ATR=4.28 |
| Stop hit — per-position SL triggered | 2024-12-12 10:25:00 | 2334.06 | 2347.28 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-12-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 11:10:00 | 2306.41 | 2308.96 | 0.00 | ORB-short ORB[2306.66,2321.46] vol=3.7x ATR=4.64 |
| Stop hit — per-position SL triggered | 2024-12-13 11:20:00 | 2311.05 | 2308.88 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-12-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 11:10:00 | 2330.61 | 2337.05 | 0.00 | ORB-short ORB[2336.81,2353.92] vol=2.7x ATR=4.30 |
| Stop hit — per-position SL triggered | 2024-12-16 11:35:00 | 2334.91 | 2336.23 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-12-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 10:05:00 | 2310.54 | 2304.18 | 0.00 | ORB-long ORB[2293.04,2310.05] vol=3.6x ATR=4.58 |
| Stop hit — per-position SL triggered | 2024-12-24 10:15:00 | 2305.96 | 2305.62 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-12-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:30:00 | 2291.46 | 2295.69 | 0.00 | ORB-short ORB[2292.45,2303.61] vol=2.2x ATR=4.09 |
| Stop hit — per-position SL triggered | 2024-12-26 10:50:00 | 2295.55 | 2294.76 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-01-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 11:05:00 | 2297.76 | 2292.07 | 0.00 | ORB-long ORB[2286.10,2296.58] vol=2.0x ATR=3.89 |
| Stop hit — per-position SL triggered | 2025-01-01 11:25:00 | 2293.87 | 2292.65 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-01-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 10:45:00 | 2303.86 | 2291.08 | 0.00 | ORB-long ORB[2280.25,2295.89] vol=1.9x ATR=4.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 11:00:00 | 2310.01 | 2295.13 | 0.00 | T1 1.5R @ 2310.01 |
| Target hit | 2025-01-02 15:20:00 | 2330.81 | 2319.32 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — SELL (started 2025-01-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 11:10:00 | 2332.19 | 2347.02 | 0.00 | ORB-short ORB[2356.58,2379.45] vol=1.5x ATR=6.08 |
| Stop hit — per-position SL triggered | 2025-01-06 12:10:00 | 2338.27 | 2342.47 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-01-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 09:55:00 | 2393.03 | 2367.96 | 0.00 | ORB-long ORB[2347.43,2374.58] vol=2.3x ATR=8.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 10:10:00 | 2405.56 | 2381.14 | 0.00 | T1 1.5R @ 2405.56 |
| Target hit | 2025-01-09 11:50:00 | 2404.44 | 2406.46 | 0.00 | Trail-exit close<VWAP |

### Cycle 59 — BUY (started 2025-01-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-10 11:00:00 | 2411.96 | 2393.91 | 0.00 | ORB-long ORB[2374.39,2407.34] vol=3.0x ATR=7.94 |
| Stop hit — per-position SL triggered | 2025-01-10 11:25:00 | 2404.02 | 2397.07 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-01-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-13 10:30:00 | 2402.71 | 2401.21 | 0.00 | ORB-long ORB[2369.71,2400.16] vol=2.5x ATR=7.00 |
| Stop hit — per-position SL triggered | 2025-01-13 11:10:00 | 2395.71 | 2402.58 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-01-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-16 10:10:00 | 2304.59 | 2311.85 | 0.00 | ORB-short ORB[2308.04,2341.14] vol=2.7x ATR=5.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-16 10:20:00 | 2296.50 | 2310.64 | 0.00 | T1 1.5R @ 2296.50 |
| Stop hit — per-position SL triggered | 2025-01-16 11:15:00 | 2304.59 | 2306.61 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-01-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 09:50:00 | 2324.41 | 2320.10 | 0.00 | ORB-long ORB[2304.74,2320.09] vol=2.0x ATR=4.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 10:25:00 | 2331.89 | 2322.07 | 0.00 | T1 1.5R @ 2331.89 |
| Target hit | 2025-01-17 12:45:00 | 2329.68 | 2330.23 | 0.00 | Trail-exit close<VWAP |

### Cycle 63 — SELL (started 2025-01-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-20 10:25:00 | 2315.46 | 2320.76 | 0.00 | ORB-short ORB[2317.97,2329.33] vol=4.2x ATR=4.38 |
| Stop hit — per-position SL triggered | 2025-01-20 10:55:00 | 2319.84 | 2319.59 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-01-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-24 10:50:00 | 2311.18 | 2299.16 | 0.00 | ORB-long ORB[2287.13,2300.56] vol=1.8x ATR=5.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 11:10:00 | 2319.31 | 2303.69 | 0.00 | T1 1.5R @ 2319.31 |
| Target hit | 2025-01-24 15:20:00 | 2326.53 | 2319.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 65 — BUY (started 2025-01-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-28 11:00:00 | 2362.53 | 2356.38 | 0.00 | ORB-long ORB[2340.60,2362.48] vol=3.0x ATR=5.86 |
| Stop hit — per-position SL triggered | 2025-01-28 11:20:00 | 2356.67 | 2356.59 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-01-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 11:10:00 | 2395.63 | 2390.25 | 0.00 | ORB-long ORB[2362.29,2389.78] vol=1.5x ATR=6.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-31 11:20:00 | 2405.75 | 2391.32 | 0.00 | T1 1.5R @ 2405.75 |
| Stop hit — per-position SL triggered | 2025-01-31 11:30:00 | 2395.63 | 2392.31 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-02-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 11:00:00 | 2347.48 | 2354.94 | 0.00 | ORB-short ORB[2354.71,2367.35] vol=1.7x ATR=4.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 11:05:00 | 2340.73 | 2354.13 | 0.00 | T1 1.5R @ 2340.73 |
| Target hit | 2025-02-06 15:20:00 | 2336.22 | 2340.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 68 — BUY (started 2025-02-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-10 09:35:00 | 2348.02 | 2340.96 | 0.00 | ORB-long ORB[2323.73,2345.07] vol=1.5x ATR=5.69 |
| Stop hit — per-position SL triggered | 2025-02-10 09:40:00 | 2342.33 | 2341.40 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-11 11:15:00 | 2310.89 | 2321.05 | 0.00 | ORB-short ORB[2324.81,2339.12] vol=2.1x ATR=4.74 |
| Stop hit — per-position SL triggered | 2025-02-11 11:20:00 | 2315.63 | 2320.84 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-13 11:15:00 | 2301.40 | 2294.72 | 0.00 | ORB-long ORB[2275.58,2300.81] vol=2.1x ATR=4.66 |
| Stop hit — per-position SL triggered | 2025-02-13 12:45:00 | 2296.74 | 2296.77 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-02-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-18 10:40:00 | 2274.54 | 2281.64 | 0.00 | ORB-short ORB[2282.61,2300.36] vol=2.8x ATR=4.88 |
| Stop hit — per-position SL triggered | 2025-02-18 10:55:00 | 2279.42 | 2281.30 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2025-02-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 10:55:00 | 2191.91 | 2200.98 | 0.00 | ORB-short ORB[2198.65,2216.75] vol=3.0x ATR=4.24 |
| Stop hit — per-position SL triggered | 2025-02-21 11:10:00 | 2196.15 | 2200.48 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2025-02-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-28 09:35:00 | 2182.52 | 2197.57 | 0.00 | ORB-short ORB[2193.68,2211.19] vol=1.7x ATR=6.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:50:00 | 2172.66 | 2190.74 | 0.00 | T1 1.5R @ 2172.66 |
| Target hit | 2025-02-28 11:50:00 | 2178.54 | 2178.24 | 0.00 | Trail-exit close>VWAP |

### Cycle 74 — SELL (started 2025-03-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-03 10:50:00 | 2126.75 | 2140.48 | 0.00 | ORB-short ORB[2135.80,2163.09] vol=2.1x ATR=5.09 |
| Stop hit — per-position SL triggered | 2025-03-03 11:15:00 | 2131.84 | 2137.37 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2025-03-04 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-04 10:40:00 | 2112.38 | 2123.86 | 0.00 | ORB-short ORB[2125.81,2144.40] vol=1.6x ATR=4.22 |
| Stop hit — per-position SL triggered | 2025-03-04 10:45:00 | 2116.60 | 2123.17 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2025-03-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-06 11:10:00 | 2164.17 | 2142.34 | 0.00 | ORB-long ORB[2135.45,2160.34] vol=2.4x ATR=5.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-06 11:35:00 | 2172.96 | 2148.31 | 0.00 | T1 1.5R @ 2172.96 |
| Target hit | 2025-03-06 15:20:00 | 2184.73 | 2168.62 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 77 — SELL (started 2025-03-07 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-07 10:25:00 | 2166.63 | 2171.11 | 0.00 | ORB-short ORB[2170.13,2190.34] vol=7.3x ATR=5.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-07 10:35:00 | 2158.38 | 2167.82 | 0.00 | T1 1.5R @ 2158.38 |
| Stop hit — per-position SL triggered | 2025-03-07 11:00:00 | 2166.63 | 2164.84 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2025-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 11:15:00 | 2218.28 | 2200.22 | 0.00 | ORB-long ORB[2158.81,2188.27] vol=1.9x ATR=4.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-10 13:20:00 | 2225.69 | 2210.13 | 0.00 | T1 1.5R @ 2225.69 |
| Stop hit — per-position SL triggered | 2025-03-10 14:20:00 | 2218.28 | 2212.93 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2025-03-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-12 10:50:00 | 2187.54 | 2179.52 | 0.00 | ORB-long ORB[2165.85,2186.55] vol=2.1x ATR=4.85 |
| Stop hit — per-position SL triggered | 2025-03-12 12:45:00 | 2182.69 | 2184.05 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2025-03-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:45:00 | 2170.86 | 2158.13 | 0.00 | ORB-long ORB[2139.48,2159.50] vol=1.7x ATR=4.85 |
| Stop hit — per-position SL triggered | 2025-03-18 10:55:00 | 2166.01 | 2159.79 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2025-03-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-19 10:45:00 | 2165.06 | 2166.78 | 0.00 | ORB-short ORB[2165.75,2176.72] vol=2.4x ATR=3.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 11:10:00 | 2159.97 | 2165.59 | 0.00 | T1 1.5R @ 2159.97 |
| Stop hit — per-position SL triggered | 2025-03-19 12:15:00 | 2165.06 | 2164.06 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2025-03-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 11:15:00 | 2208.88 | 2200.69 | 0.00 | ORB-long ORB[2199.49,2208.34] vol=1.7x ATR=3.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 11:30:00 | 2214.35 | 2202.37 | 0.00 | T1 1.5R @ 2214.35 |
| Stop hit — per-position SL triggered | 2025-03-21 11:50:00 | 2208.88 | 2204.54 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2025-04-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-04 09:35:00 | 2218.97 | 2206.40 | 0.00 | ORB-long ORB[2185.72,2209.33] vol=3.2x ATR=6.23 |
| Stop hit — per-position SL triggered | 2025-04-04 09:45:00 | 2212.74 | 2207.81 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2025-04-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-09 11:00:00 | 2301.79 | 2282.69 | 0.00 | ORB-long ORB[2252.11,2280.15] vol=1.8x ATR=6.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-09 11:50:00 | 2311.21 | 2290.75 | 0.00 | T1 1.5R @ 2311.21 |
| Stop hit — per-position SL triggered | 2025-04-09 12:00:00 | 2301.79 | 2291.44 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2025-04-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-21 10:00:00 | 2315.76 | 2319.14 | 0.00 | ORB-short ORB[2317.14,2333.66] vol=1.8x ATR=5.45 |
| Target hit | 2025-04-21 15:20:00 | 2312.12 | 2316.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 86 — BUY (started 2025-04-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 09:40:00 | 2335.83 | 2327.24 | 0.00 | ORB-long ORB[2316.64,2331.30] vol=1.6x ATR=4.66 |
| Stop hit — per-position SL triggered | 2025-04-22 09:50:00 | 2331.17 | 2328.38 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2025-04-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 09:45:00 | 2277.49 | 2281.71 | 0.00 | ORB-short ORB[2278.28,2294.41] vol=4.8x ATR=4.25 |
| Stop hit — per-position SL triggered | 2025-04-29 10:00:00 | 2281.74 | 2281.43 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2025-05-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 11:05:00 | 2311.63 | 2306.01 | 0.00 | ORB-long ORB[2281.23,2303.56] vol=2.3x ATR=4.29 |
| Stop hit — per-position SL triggered | 2025-05-05 11:30:00 | 2307.34 | 2308.23 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2025-05-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-06 10:45:00 | 2335.63 | 2325.38 | 0.00 | ORB-long ORB[2301.99,2321.46] vol=2.2x ATR=4.57 |
| Stop hit — per-position SL triggered | 2025-05-06 12:00:00 | 2331.06 | 2328.71 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-15 10:30:00 | 2291.95 | 2024-05-15 11:00:00 | 2295.48 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2024-05-16 09:30:00 | 2277.79 | 2024-05-16 10:00:00 | 2270.85 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-05-16 09:30:00 | 2277.79 | 2024-05-16 11:35:00 | 2274.30 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2024-05-22 09:40:00 | 2306.76 | 2024-05-22 09:45:00 | 2302.58 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-05-27 10:25:00 | 2355.15 | 2024-05-27 11:00:00 | 2349.24 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-05-28 09:30:00 | 2341.28 | 2024-05-28 09:45:00 | 2346.57 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-05-30 10:40:00 | 2319.45 | 2024-05-30 12:00:00 | 2324.32 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-06-04 09:40:00 | 2350.48 | 2024-06-04 09:45:00 | 2365.58 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-06-04 09:40:00 | 2350.48 | 2024-06-04 15:20:00 | 2456.62 | TARGET_HIT | 0.50 | 4.52% |
| SELL | retest1 | 2024-06-12 11:15:00 | 2487.95 | 2024-06-12 11:20:00 | 2493.29 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-06-19 10:50:00 | 2437.54 | 2024-06-19 10:55:00 | 2442.20 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-06-21 10:45:00 | 2403.50 | 2024-06-21 11:30:00 | 2396.19 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-06-21 10:45:00 | 2403.50 | 2024-06-21 13:20:00 | 2403.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-26 10:15:00 | 2413.63 | 2024-06-26 10:25:00 | 2419.90 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2024-06-26 10:15:00 | 2413.63 | 2024-06-26 10:50:00 | 2413.63 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-27 10:05:00 | 2425.14 | 2024-06-27 10:15:00 | 2419.65 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-07-01 09:35:00 | 2452.39 | 2024-07-01 09:50:00 | 2462.96 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-07-01 09:35:00 | 2452.39 | 2024-07-01 15:20:00 | 2459.72 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2024-07-02 10:30:00 | 2449.34 | 2024-07-02 12:45:00 | 2441.45 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-07-02 10:30:00 | 2449.34 | 2024-07-02 12:50:00 | 2449.34 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-04 10:40:00 | 2483.47 | 2024-07-04 10:50:00 | 2492.58 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-07-04 10:40:00 | 2483.47 | 2024-07-04 11:05:00 | 2483.47 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-05 11:10:00 | 2479.34 | 2024-07-05 12:05:00 | 2485.25 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2024-07-05 11:10:00 | 2479.34 | 2024-07-05 15:20:00 | 2506.20 | TARGET_HIT | 0.50 | 1.08% |
| BUY | retest1 | 2024-07-08 10:50:00 | 2546.48 | 2024-07-08 10:55:00 | 2538.31 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-07-10 10:40:00 | 2538.12 | 2024-07-10 11:50:00 | 2545.25 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-12 10:50:00 | 2577.22 | 2024-07-12 10:55:00 | 2584.89 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-07-12 10:50:00 | 2577.22 | 2024-07-12 11:15:00 | 2577.22 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-16 11:05:00 | 2613.66 | 2024-07-16 11:10:00 | 2623.16 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-07-16 11:05:00 | 2613.66 | 2024-07-16 15:20:00 | 2642.98 | TARGET_HIT | 0.50 | 1.12% |
| BUY | retest1 | 2024-07-23 11:10:00 | 2717.49 | 2024-07-23 11:15:00 | 2711.52 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-07-25 10:50:00 | 2648.04 | 2024-07-25 11:10:00 | 2637.79 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-07-25 10:50:00 | 2648.04 | 2024-07-25 12:40:00 | 2648.04 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-26 10:55:00 | 2646.03 | 2024-07-26 11:00:00 | 2651.76 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-07-31 10:45:00 | 2637.62 | 2024-07-31 11:35:00 | 2642.75 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-08-01 10:25:00 | 2661.08 | 2024-08-01 11:10:00 | 2652.79 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-08-01 10:25:00 | 2661.08 | 2024-08-01 12:30:00 | 2661.08 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-21 10:10:00 | 2726.49 | 2024-08-21 10:35:00 | 2721.55 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-08-22 10:45:00 | 2742.47 | 2024-08-22 13:15:00 | 2747.96 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-08-26 10:50:00 | 2786.15 | 2024-08-26 11:05:00 | 2781.14 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-08-28 10:50:00 | 2710.01 | 2024-08-28 10:55:00 | 2703.92 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2024-08-28 10:50:00 | 2710.01 | 2024-08-28 11:45:00 | 2710.01 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-29 10:45:00 | 2738.15 | 2024-08-29 11:15:00 | 2745.56 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2024-08-29 10:45:00 | 2738.15 | 2024-08-29 11:25:00 | 2738.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-02 10:50:00 | 2755.02 | 2024-09-02 11:55:00 | 2749.30 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-09-03 09:30:00 | 2757.23 | 2024-09-03 09:40:00 | 2765.99 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-09-03 09:30:00 | 2757.23 | 2024-09-03 09:45:00 | 2757.23 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-04 10:05:00 | 2767.51 | 2024-09-04 10:55:00 | 2776.80 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-09-04 10:05:00 | 2767.51 | 2024-09-04 15:20:00 | 2794.81 | TARGET_HIT | 0.50 | 0.99% |
| BUY | retest1 | 2024-09-05 09:30:00 | 2801.10 | 2024-09-05 09:55:00 | 2795.39 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-09-11 10:50:00 | 2886.34 | 2024-09-11 11:00:00 | 2894.21 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2024-09-11 10:50:00 | 2886.34 | 2024-09-11 11:35:00 | 2886.34 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-12 10:25:00 | 2870.70 | 2024-09-12 10:30:00 | 2864.33 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-09-19 09:35:00 | 2844.78 | 2024-09-19 09:55:00 | 2853.00 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2024-09-19 09:35:00 | 2844.78 | 2024-09-19 12:35:00 | 2853.97 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2024-10-07 11:00:00 | 2806.91 | 2024-10-07 11:15:00 | 2798.15 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-10-09 10:50:00 | 2745.97 | 2024-10-09 11:10:00 | 2755.53 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-10-10 11:10:00 | 2700.62 | 2024-10-10 12:05:00 | 2689.34 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-10-10 11:10:00 | 2700.62 | 2024-10-10 13:10:00 | 2700.62 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-15 10:25:00 | 2724.13 | 2024-10-15 10:50:00 | 2730.31 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-10-17 11:05:00 | 2712.96 | 2024-10-17 11:35:00 | 2719.56 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-10-29 10:45:00 | 2507.18 | 2024-10-29 10:50:00 | 2513.44 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-11-04 10:00:00 | 2484.75 | 2024-11-04 10:10:00 | 2475.40 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-11-04 10:00:00 | 2484.75 | 2024-11-04 12:00:00 | 2484.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-08 10:30:00 | 2449.59 | 2024-11-08 10:55:00 | 2457.51 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-11-08 10:30:00 | 2449.59 | 2024-11-08 13:35:00 | 2449.93 | TARGET_HIT | 0.50 | 0.01% |
| SELL | retest1 | 2024-11-12 10:50:00 | 2436.95 | 2024-11-12 10:55:00 | 2442.23 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-11-14 11:05:00 | 2388.40 | 2024-11-14 11:50:00 | 2378.24 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-11-14 11:05:00 | 2388.40 | 2024-11-14 15:20:00 | 2347.53 | TARGET_HIT | 0.50 | 1.71% |
| SELL | retest1 | 2024-12-05 10:55:00 | 2412.11 | 2024-12-05 11:05:00 | 2417.41 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-06 10:45:00 | 2440.49 | 2024-12-06 15:20:00 | 2442.31 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest1 | 2024-12-12 10:20:00 | 2329.78 | 2024-12-12 10:25:00 | 2334.06 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-12-13 11:10:00 | 2306.41 | 2024-12-13 11:20:00 | 2311.05 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-12-16 11:10:00 | 2330.61 | 2024-12-16 11:35:00 | 2334.91 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-12-24 10:05:00 | 2310.54 | 2024-12-24 10:15:00 | 2305.96 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-12-26 10:30:00 | 2291.46 | 2024-12-26 10:50:00 | 2295.55 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-01-01 11:05:00 | 2297.76 | 2025-01-01 11:25:00 | 2293.87 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-01-02 10:45:00 | 2303.86 | 2025-01-02 11:00:00 | 2310.01 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2025-01-02 10:45:00 | 2303.86 | 2025-01-02 15:20:00 | 2330.81 | TARGET_HIT | 0.50 | 1.17% |
| SELL | retest1 | 2025-01-06 11:10:00 | 2332.19 | 2025-01-06 12:10:00 | 2338.27 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-01-09 09:55:00 | 2393.03 | 2025-01-09 10:10:00 | 2405.56 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-01-09 09:55:00 | 2393.03 | 2025-01-09 11:50:00 | 2404.44 | TARGET_HIT | 0.50 | 0.48% |
| BUY | retest1 | 2025-01-10 11:00:00 | 2411.96 | 2025-01-10 11:25:00 | 2404.02 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-01-13 10:30:00 | 2402.71 | 2025-01-13 11:10:00 | 2395.71 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-01-16 10:10:00 | 2304.59 | 2025-01-16 10:20:00 | 2296.50 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-01-16 10:10:00 | 2304.59 | 2025-01-16 11:15:00 | 2304.59 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-17 09:50:00 | 2324.41 | 2025-01-17 10:25:00 | 2331.89 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-01-17 09:50:00 | 2324.41 | 2025-01-17 12:45:00 | 2329.68 | TARGET_HIT | 0.50 | 0.23% |
| SELL | retest1 | 2025-01-20 10:25:00 | 2315.46 | 2025-01-20 10:55:00 | 2319.84 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-01-24 10:50:00 | 2311.18 | 2025-01-24 11:10:00 | 2319.31 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-01-24 10:50:00 | 2311.18 | 2025-01-24 15:20:00 | 2326.53 | TARGET_HIT | 0.50 | 0.66% |
| BUY | retest1 | 2025-01-28 11:00:00 | 2362.53 | 2025-01-28 11:20:00 | 2356.67 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-01-31 11:10:00 | 2395.63 | 2025-01-31 11:20:00 | 2405.75 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-01-31 11:10:00 | 2395.63 | 2025-01-31 11:30:00 | 2395.63 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-06 11:00:00 | 2347.48 | 2025-02-06 11:05:00 | 2340.73 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-02-06 11:00:00 | 2347.48 | 2025-02-06 15:20:00 | 2336.22 | TARGET_HIT | 0.50 | 0.48% |
| BUY | retest1 | 2025-02-10 09:35:00 | 2348.02 | 2025-02-10 09:40:00 | 2342.33 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-02-11 11:15:00 | 2310.89 | 2025-02-11 11:20:00 | 2315.63 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-02-13 11:15:00 | 2301.40 | 2025-02-13 12:45:00 | 2296.74 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-02-18 10:40:00 | 2274.54 | 2025-02-18 10:55:00 | 2279.42 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-02-21 10:55:00 | 2191.91 | 2025-02-21 11:10:00 | 2196.15 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-02-28 09:35:00 | 2182.52 | 2025-02-28 09:50:00 | 2172.66 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-02-28 09:35:00 | 2182.52 | 2025-02-28 11:50:00 | 2178.54 | TARGET_HIT | 0.50 | 0.18% |
| SELL | retest1 | 2025-03-03 10:50:00 | 2126.75 | 2025-03-03 11:15:00 | 2131.84 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-03-04 10:40:00 | 2112.38 | 2025-03-04 10:45:00 | 2116.60 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-03-06 11:10:00 | 2164.17 | 2025-03-06 11:35:00 | 2172.96 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-03-06 11:10:00 | 2164.17 | 2025-03-06 15:20:00 | 2184.73 | TARGET_HIT | 0.50 | 0.95% |
| SELL | retest1 | 2025-03-07 10:25:00 | 2166.63 | 2025-03-07 10:35:00 | 2158.38 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-03-07 10:25:00 | 2166.63 | 2025-03-07 11:00:00 | 2166.63 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-10 11:15:00 | 2218.28 | 2025-03-10 13:20:00 | 2225.69 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-03-10 11:15:00 | 2218.28 | 2025-03-10 14:20:00 | 2218.28 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-12 10:50:00 | 2187.54 | 2025-03-12 12:45:00 | 2182.69 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-03-18 10:45:00 | 2170.86 | 2025-03-18 10:55:00 | 2166.01 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-03-19 10:45:00 | 2165.06 | 2025-03-19 11:10:00 | 2159.97 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2025-03-19 10:45:00 | 2165.06 | 2025-03-19 12:15:00 | 2165.06 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-21 11:15:00 | 2208.88 | 2025-03-21 11:30:00 | 2214.35 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2025-03-21 11:15:00 | 2208.88 | 2025-03-21 11:50:00 | 2208.88 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-04 09:35:00 | 2218.97 | 2025-04-04 09:45:00 | 2212.74 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-04-09 11:00:00 | 2301.79 | 2025-04-09 11:50:00 | 2311.21 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-04-09 11:00:00 | 2301.79 | 2025-04-09 12:00:00 | 2301.79 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-21 10:00:00 | 2315.76 | 2025-04-21 15:20:00 | 2312.12 | TARGET_HIT | 1.00 | 0.16% |
| BUY | retest1 | 2025-04-22 09:40:00 | 2335.83 | 2025-04-22 09:50:00 | 2331.17 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-04-29 09:45:00 | 2277.49 | 2025-04-29 10:00:00 | 2281.74 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-05-05 11:05:00 | 2311.63 | 2025-05-05 11:30:00 | 2307.34 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-05-06 10:45:00 | 2335.63 | 2025-05-06 12:00:00 | 2331.06 | STOP_HIT | 1.00 | -0.20% |
