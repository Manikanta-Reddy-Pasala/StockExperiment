# GRASIM (GRASIM)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 2965.00
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
| ENTRY1 | 106 |
| ENTRY2 | 0 |
| PARTIAL | 45 |
| TARGET_HIT | 17 |
| STOP_HIT | 89 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 151 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 62 / 89
- **Target hits / Stop hits / Partials:** 17 / 89 / 45
- **Avg / median % per leg:** 0.13% / 0.00%
- **Sum % (uncompounded):** 18.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 74 | 27 | 36.5% | 6 | 47 | 21 | 0.10% | 7.7% |
| BUY @ 2nd Alert (retest1) | 74 | 27 | 36.5% | 6 | 47 | 21 | 0.10% | 7.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 77 | 35 | 45.5% | 11 | 42 | 24 | 0.15% | 11.3% |
| SELL @ 2nd Alert (retest1) | 77 | 35 | 45.5% | 11 | 42 | 24 | 0.15% | 11.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 151 | 62 | 41.1% | 17 | 89 | 45 | 0.13% | 19.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 10:55:00 | 2349.95 | 2363.08 | 0.00 | ORB-short ORB[2365.70,2393.00] vol=1.9x ATR=10.71 |
| Stop hit — per-position SL triggered | 2024-05-13 11:15:00 | 2360.66 | 2361.48 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-14 11:05:00 | 2370.10 | 2377.91 | 0.00 | ORB-short ORB[2380.00,2396.85] vol=1.7x ATR=5.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-14 11:25:00 | 2362.60 | 2376.07 | 0.00 | T1 1.5R @ 2362.60 |
| Stop hit — per-position SL triggered | 2024-05-14 11:35:00 | 2370.10 | 2374.60 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 10:20:00 | 2351.60 | 2371.23 | 0.00 | ORB-short ORB[2369.10,2385.85] vol=2.9x ATR=6.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 10:45:00 | 2341.13 | 2364.81 | 0.00 | T1 1.5R @ 2341.13 |
| Stop hit — per-position SL triggered | 2024-05-16 12:10:00 | 2351.60 | 2357.02 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 10:55:00 | 2417.75 | 2406.37 | 0.00 | ORB-long ORB[2374.00,2409.95] vol=1.9x ATR=7.93 |
| Stop hit — per-position SL triggered | 2024-05-17 12:30:00 | 2409.82 | 2410.83 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 10:55:00 | 2431.55 | 2437.57 | 0.00 | ORB-short ORB[2442.60,2463.05] vol=2.2x ATR=7.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-22 11:05:00 | 2421.05 | 2434.92 | 0.00 | T1 1.5R @ 2421.05 |
| Stop hit — per-position SL triggered | 2024-05-22 11:30:00 | 2431.55 | 2432.46 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 10:35:00 | 2414.15 | 2428.76 | 0.00 | ORB-short ORB[2424.55,2455.70] vol=3.0x ATR=7.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 10:50:00 | 2403.05 | 2423.62 | 0.00 | T1 1.5R @ 2403.05 |
| Target hit | 2024-05-27 15:20:00 | 2390.35 | 2404.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2024-05-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 10:40:00 | 2441.55 | 2428.87 | 0.00 | ORB-long ORB[2421.00,2441.50] vol=1.9x ATR=7.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-29 11:20:00 | 2452.19 | 2435.51 | 0.00 | T1 1.5R @ 2452.19 |
| Stop hit — per-position SL triggered | 2024-05-29 11:30:00 | 2441.55 | 2436.38 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 09:45:00 | 2405.05 | 2391.69 | 0.00 | ORB-long ORB[2375.30,2390.95] vol=2.0x ATR=7.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 10:00:00 | 2416.90 | 2401.73 | 0.00 | T1 1.5R @ 2416.90 |
| Target hit | 2024-06-10 15:10:00 | 2445.00 | 2446.01 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2024-06-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 09:30:00 | 2455.25 | 2448.69 | 0.00 | ORB-long ORB[2430.00,2454.50] vol=1.7x ATR=7.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 09:45:00 | 2466.39 | 2451.86 | 0.00 | T1 1.5R @ 2466.39 |
| Stop hit — per-position SL triggered | 2024-06-12 10:05:00 | 2455.25 | 2453.45 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 10:50:00 | 2484.05 | 2492.64 | 0.00 | ORB-short ORB[2487.05,2515.60] vol=1.8x ATR=8.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-21 12:25:00 | 2472.03 | 2489.46 | 0.00 | T1 1.5R @ 2472.03 |
| Stop hit — per-position SL triggered | 2024-06-21 14:50:00 | 2484.05 | 2481.22 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 09:30:00 | 2537.25 | 2526.38 | 0.00 | ORB-long ORB[2509.05,2535.75] vol=1.8x ATR=7.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 09:55:00 | 2547.91 | 2534.87 | 0.00 | T1 1.5R @ 2547.91 |
| Stop hit — per-position SL triggered | 2024-06-26 10:30:00 | 2537.25 | 2545.72 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-02 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 10:20:00 | 2757.55 | 2744.50 | 0.00 | ORB-long ORB[2717.00,2742.00] vol=1.6x ATR=7.64 |
| Stop hit — per-position SL triggered | 2024-07-02 12:05:00 | 2749.91 | 2748.94 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 09:30:00 | 2753.80 | 2747.40 | 0.00 | ORB-long ORB[2733.05,2753.40] vol=1.6x ATR=7.89 |
| Stop hit — per-position SL triggered | 2024-07-03 13:05:00 | 2745.91 | 2753.41 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 10:30:00 | 2718.15 | 2734.76 | 0.00 | ORB-short ORB[2740.00,2752.75] vol=2.0x ATR=6.31 |
| Stop hit — per-position SL triggered | 2024-07-04 10:35:00 | 2724.46 | 2733.79 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 10:40:00 | 2833.00 | 2821.42 | 0.00 | ORB-long ORB[2802.00,2826.80] vol=1.8x ATR=7.93 |
| Stop hit — per-position SL triggered | 2024-07-11 10:55:00 | 2825.07 | 2822.25 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 10:45:00 | 2830.45 | 2820.24 | 0.00 | ORB-long ORB[2801.60,2822.00] vol=2.8x ATR=6.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 10:55:00 | 2839.77 | 2823.91 | 0.00 | T1 1.5R @ 2839.77 |
| Stop hit — per-position SL triggered | 2024-07-12 12:20:00 | 2830.45 | 2831.89 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 10:05:00 | 2828.55 | 2823.12 | 0.00 | ORB-long ORB[2810.30,2825.45] vol=2.1x ATR=5.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 10:20:00 | 2837.13 | 2828.62 | 0.00 | T1 1.5R @ 2837.13 |
| Stop hit — per-position SL triggered | 2024-07-16 11:20:00 | 2828.55 | 2830.60 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 11:15:00 | 2799.50 | 2821.24 | 0.00 | ORB-short ORB[2810.00,2845.80] vol=7.3x ATR=7.13 |
| Stop hit — per-position SL triggered | 2024-07-18 11:20:00 | 2806.63 | 2820.60 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:55:00 | 2856.80 | 2828.28 | 0.00 | ORB-long ORB[2810.30,2825.00] vol=3.2x ATR=7.31 |
| Stop hit — per-position SL triggered | 2024-07-26 11:00:00 | 2849.49 | 2833.43 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-07-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-29 10:50:00 | 2831.20 | 2834.65 | 0.00 | ORB-short ORB[2840.75,2865.00] vol=4.1x ATR=6.42 |
| Stop hit — per-position SL triggered | 2024-07-29 11:00:00 | 2837.62 | 2834.65 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-02 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-02 10:35:00 | 2734.80 | 2742.16 | 0.00 | ORB-short ORB[2740.00,2756.60] vol=2.2x ATR=7.49 |
| Stop hit — per-position SL triggered | 2024-08-02 10:55:00 | 2742.29 | 2741.08 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-05 11:10:00 | 2629.20 | 2652.78 | 0.00 | ORB-short ORB[2646.70,2678.95] vol=2.0x ATR=9.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 11:20:00 | 2615.42 | 2649.64 | 0.00 | T1 1.5R @ 2615.42 |
| Stop hit — per-position SL triggered | 2024-08-05 12:00:00 | 2629.20 | 2633.16 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-06 10:20:00 | 2659.35 | 2648.29 | 0.00 | ORB-long ORB[2623.00,2647.00] vol=2.2x ATR=7.06 |
| Stop hit — per-position SL triggered | 2024-08-06 10:30:00 | 2652.29 | 2650.01 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-08 11:00:00 | 2597.55 | 2607.97 | 0.00 | ORB-short ORB[2611.00,2633.80] vol=2.3x ATR=7.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-08 11:10:00 | 2586.30 | 2606.52 | 0.00 | T1 1.5R @ 2586.30 |
| Stop hit — per-position SL triggered | 2024-08-08 11:35:00 | 2597.55 | 2602.98 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 10:15:00 | 2535.35 | 2557.06 | 0.00 | ORB-short ORB[2554.80,2579.40] vol=1.6x ATR=8.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 12:40:00 | 2522.60 | 2543.63 | 0.00 | T1 1.5R @ 2522.60 |
| Stop hit — per-position SL triggered | 2024-08-13 13:10:00 | 2535.35 | 2536.93 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 10:00:00 | 2620.35 | 2607.92 | 0.00 | ORB-long ORB[2589.05,2613.00] vol=1.5x ATR=7.27 |
| Stop hit — per-position SL triggered | 2024-08-20 11:00:00 | 2613.08 | 2616.38 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 10:45:00 | 2651.05 | 2639.54 | 0.00 | ORB-long ORB[2625.15,2644.00] vol=2.0x ATR=6.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 14:00:00 | 2660.21 | 2646.34 | 0.00 | T1 1.5R @ 2660.21 |
| Target hit | 2024-08-21 15:20:00 | 2691.65 | 2661.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — SELL (started 2024-08-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:50:00 | 2674.85 | 2690.35 | 0.00 | ORB-short ORB[2694.85,2719.00] vol=1.8x ATR=6.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 11:05:00 | 2664.93 | 2685.19 | 0.00 | T1 1.5R @ 2664.93 |
| Stop hit — per-position SL triggered | 2024-08-29 11:25:00 | 2674.85 | 2683.04 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-09-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 09:55:00 | 2724.65 | 2713.21 | 0.00 | ORB-long ORB[2687.25,2714.95] vol=1.6x ATR=7.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 10:05:00 | 2736.22 | 2719.64 | 0.00 | T1 1.5R @ 2736.22 |
| Stop hit — per-position SL triggered | 2024-09-04 10:40:00 | 2724.65 | 2725.91 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-09-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 11:05:00 | 2696.55 | 2713.47 | 0.00 | ORB-short ORB[2722.75,2744.60] vol=1.7x ATR=7.18 |
| Stop hit — per-position SL triggered | 2024-09-06 11:25:00 | 2703.73 | 2712.07 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 10:00:00 | 2692.60 | 2699.78 | 0.00 | ORB-short ORB[2694.10,2715.15] vol=1.7x ATR=6.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 10:25:00 | 2682.39 | 2696.77 | 0.00 | T1 1.5R @ 2682.39 |
| Stop hit — per-position SL triggered | 2024-09-10 11:15:00 | 2692.60 | 2692.59 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 10:50:00 | 2719.35 | 2705.73 | 0.00 | ORB-long ORB[2687.30,2712.00] vol=5.0x ATR=6.17 |
| Stop hit — per-position SL triggered | 2024-09-11 11:15:00 | 2713.18 | 2710.03 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-09-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 11:05:00 | 2788.30 | 2802.93 | 0.00 | ORB-short ORB[2795.00,2819.05] vol=1.5x ATR=7.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-16 13:35:00 | 2777.20 | 2795.98 | 0.00 | T1 1.5R @ 2777.20 |
| Target hit | 2024-09-16 15:20:00 | 2767.30 | 2787.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — BUY (started 2024-09-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 09:55:00 | 2764.10 | 2754.35 | 0.00 | ORB-long ORB[2728.05,2759.85] vol=3.1x ATR=6.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 10:10:00 | 2774.16 | 2757.97 | 0.00 | T1 1.5R @ 2774.16 |
| Stop hit — per-position SL triggered | 2024-09-18 10:20:00 | 2764.10 | 2762.37 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-09-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-20 10:25:00 | 2706.20 | 2719.17 | 0.00 | ORB-short ORB[2738.30,2755.70] vol=2.1x ATR=8.28 |
| Stop hit — per-position SL triggered | 2024-09-20 10:45:00 | 2714.48 | 2714.42 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-09-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 11:10:00 | 2596.00 | 2600.62 | 0.00 | ORB-short ORB[2605.00,2623.00] vol=1.9x ATR=6.77 |
| Stop hit — per-position SL triggered | 2024-09-25 11:15:00 | 2602.77 | 2601.34 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-09-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 09:45:00 | 2790.10 | 2776.37 | 0.00 | ORB-long ORB[2745.85,2773.00] vol=2.1x ATR=10.37 |
| Stop hit — per-position SL triggered | 2024-09-27 10:20:00 | 2779.73 | 2780.34 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-09-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-30 10:20:00 | 2816.25 | 2796.09 | 0.00 | ORB-long ORB[2768.05,2789.15] vol=1.8x ATR=10.62 |
| Stop hit — per-position SL triggered | 2024-09-30 10:30:00 | 2805.63 | 2798.39 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-10-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-03 09:50:00 | 2815.70 | 2795.73 | 0.00 | ORB-long ORB[2769.45,2795.00] vol=1.7x ATR=9.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 10:05:00 | 2829.64 | 2809.45 | 0.00 | T1 1.5R @ 2829.64 |
| Stop hit — per-position SL triggered | 2024-10-03 10:15:00 | 2815.70 | 2812.10 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-10-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:35:00 | 2736.75 | 2748.92 | 0.00 | ORB-short ORB[2758.45,2769.95] vol=1.8x ATR=8.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 11:00:00 | 2724.06 | 2740.47 | 0.00 | T1 1.5R @ 2724.06 |
| Stop hit — per-position SL triggered | 2024-10-07 11:05:00 | 2736.75 | 2739.04 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 11:15:00 | 2731.00 | 2720.98 | 0.00 | ORB-long ORB[2705.20,2726.75] vol=3.4x ATR=7.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 11:35:00 | 2742.19 | 2723.89 | 0.00 | T1 1.5R @ 2742.19 |
| Stop hit — per-position SL triggered | 2024-10-08 12:05:00 | 2731.00 | 2726.43 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-10-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 10:50:00 | 2733.00 | 2726.40 | 0.00 | ORB-long ORB[2715.85,2730.05] vol=1.5x ATR=7.16 |
| Stop hit — per-position SL triggered | 2024-10-10 11:05:00 | 2725.84 | 2726.80 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-10-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 11:05:00 | 2693.60 | 2706.77 | 0.00 | ORB-short ORB[2706.00,2730.95] vol=1.7x ATR=5.98 |
| Stop hit — per-position SL triggered | 2024-10-11 11:50:00 | 2699.58 | 2700.09 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-10-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 09:45:00 | 2719.10 | 2724.88 | 0.00 | ORB-short ORB[2724.20,2737.55] vol=2.0x ATR=6.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 10:55:00 | 2709.69 | 2721.58 | 0.00 | T1 1.5R @ 2709.69 |
| Target hit | 2024-10-14 13:40:00 | 2716.40 | 2715.22 | 0.00 | Trail-exit close>VWAP |

### Cycle 45 — BUY (started 2024-10-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 09:30:00 | 2745.30 | 2735.70 | 0.00 | ORB-long ORB[2717.20,2739.75] vol=1.6x ATR=6.24 |
| Stop hit — per-position SL triggered | 2024-10-15 09:40:00 | 2739.06 | 2736.84 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-10-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 10:05:00 | 2756.95 | 2744.08 | 0.00 | ORB-long ORB[2725.00,2737.15] vol=1.6x ATR=7.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 10:10:00 | 2767.67 | 2749.23 | 0.00 | T1 1.5R @ 2767.67 |
| Stop hit — per-position SL triggered | 2024-10-16 10:15:00 | 2756.95 | 2751.07 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-10-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 09:35:00 | 2762.00 | 2776.86 | 0.00 | ORB-short ORB[2772.10,2789.80] vol=2.2x ATR=8.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 09:45:00 | 2749.71 | 2770.06 | 0.00 | T1 1.5R @ 2749.71 |
| Stop hit — per-position SL triggered | 2024-10-17 10:25:00 | 2762.00 | 2763.42 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-10-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-18 10:25:00 | 2725.00 | 2704.85 | 0.00 | ORB-long ORB[2674.65,2704.95] vol=1.6x ATR=10.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 12:05:00 | 2740.08 | 2716.40 | 0.00 | T1 1.5R @ 2740.08 |
| Target hit | 2024-10-18 15:20:00 | 2761.80 | 2735.62 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2024-10-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:40:00 | 2693.75 | 2715.89 | 0.00 | ORB-short ORB[2718.10,2746.15] vol=1.5x ATR=9.83 |
| Stop hit — per-position SL triggered | 2024-10-22 10:45:00 | 2703.58 | 2714.47 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-10-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-23 09:55:00 | 2633.35 | 2644.74 | 0.00 | ORB-short ORB[2635.00,2665.00] vol=1.7x ATR=12.53 |
| Stop hit — per-position SL triggered | 2024-10-23 10:05:00 | 2645.88 | 2643.19 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-10-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-24 09:45:00 | 2670.20 | 2643.10 | 0.00 | ORB-long ORB[2616.10,2655.05] vol=1.8x ATR=11.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-24 10:05:00 | 2686.82 | 2658.73 | 0.00 | T1 1.5R @ 2686.82 |
| Stop hit — per-position SL triggered | 2024-10-24 10:45:00 | 2670.20 | 2667.17 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-10-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:50:00 | 2623.45 | 2642.38 | 0.00 | ORB-short ORB[2650.40,2675.05] vol=2.1x ATR=7.98 |
| Stop hit — per-position SL triggered | 2024-10-25 10:55:00 | 2631.43 | 2638.54 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-10-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-28 09:40:00 | 2595.80 | 2612.34 | 0.00 | ORB-short ORB[2600.00,2634.05] vol=1.6x ATR=9.34 |
| Stop hit — per-position SL triggered | 2024-10-28 09:45:00 | 2605.14 | 2610.87 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-10-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-29 10:00:00 | 2676.00 | 2663.25 | 0.00 | ORB-long ORB[2651.10,2669.90] vol=1.7x ATR=8.39 |
| Stop hit — per-position SL triggered | 2024-10-29 10:05:00 | 2667.61 | 2664.05 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-10-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 11:05:00 | 2699.65 | 2690.60 | 0.00 | ORB-long ORB[2668.05,2694.95] vol=1.8x ATR=6.00 |
| Stop hit — per-position SL triggered | 2024-10-30 11:15:00 | 2693.65 | 2691.13 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-11-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-04 09:55:00 | 2664.60 | 2668.85 | 0.00 | ORB-short ORB[2673.60,2700.00] vol=2.3x ATR=10.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-04 10:10:00 | 2648.34 | 2666.07 | 0.00 | T1 1.5R @ 2648.34 |
| Target hit | 2024-11-04 15:20:00 | 2592.75 | 2605.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — SELL (started 2024-11-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-06 11:10:00 | 2622.30 | 2645.40 | 0.00 | ORB-short ORB[2634.30,2666.00] vol=2.6x ATR=7.38 |
| Stop hit — per-position SL triggered | 2024-11-06 11:30:00 | 2629.68 | 2644.61 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-11-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:45:00 | 2494.05 | 2508.68 | 0.00 | ORB-short ORB[2509.35,2534.65] vol=1.8x ATR=8.87 |
| Stop hit — per-position SL triggered | 2024-11-13 09:50:00 | 2502.92 | 2508.74 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-11-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:30:00 | 2600.35 | 2618.91 | 0.00 | ORB-short ORB[2608.60,2635.95] vol=1.7x ATR=7.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 10:40:00 | 2589.72 | 2609.80 | 0.00 | T1 1.5R @ 2589.72 |
| Stop hit — per-position SL triggered | 2024-11-28 10:45:00 | 2600.35 | 2608.32 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2024-11-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 11:00:00 | 2607.25 | 2597.81 | 0.00 | ORB-long ORB[2559.00,2589.60] vol=4.2x ATR=8.48 |
| Stop hit — per-position SL triggered | 2024-11-29 11:15:00 | 2598.77 | 2597.99 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2024-12-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 11:00:00 | 2677.85 | 2658.55 | 0.00 | ORB-long ORB[2591.75,2626.70] vol=6.0x ATR=8.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-02 11:30:00 | 2690.53 | 2666.82 | 0.00 | T1 1.5R @ 2690.53 |
| Target hit | 2024-12-02 15:20:00 | 2691.20 | 2680.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 2680.35 | 2697.86 | 0.00 | ORB-short ORB[2702.05,2733.00] vol=2.7x ATR=7.72 |
| Stop hit — per-position SL triggered | 2024-12-05 12:05:00 | 2688.07 | 2693.76 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2024-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-09 10:15:00 | 2671.85 | 2687.65 | 0.00 | ORB-short ORB[2697.15,2714.05] vol=2.2x ATR=5.99 |
| Stop hit — per-position SL triggered | 2024-12-09 10:30:00 | 2677.84 | 2683.74 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2024-12-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 11:10:00 | 2652.75 | 2657.93 | 0.00 | ORB-short ORB[2657.80,2683.00] vol=2.0x ATR=5.07 |
| Stop hit — per-position SL triggered | 2024-12-12 11:35:00 | 2657.82 | 2657.50 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2024-12-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-18 11:10:00 | 2590.00 | 2603.77 | 0.00 | ORB-short ORB[2591.65,2612.90] vol=1.7x ATR=6.27 |
| Stop hit — per-position SL triggered | 2024-12-18 11:25:00 | 2596.27 | 2603.20 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2024-12-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-23 11:00:00 | 2544.80 | 2523.84 | 0.00 | ORB-long ORB[2495.00,2518.95] vol=1.7x ATR=7.01 |
| Stop hit — per-position SL triggered | 2024-12-23 11:55:00 | 2537.79 | 2535.12 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2024-12-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:30:00 | 2495.40 | 2504.36 | 0.00 | ORB-short ORB[2499.75,2518.25] vol=1.8x ATR=7.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 12:30:00 | 2484.82 | 2498.06 | 0.00 | T1 1.5R @ 2484.82 |
| Target hit | 2024-12-26 14:35:00 | 2492.20 | 2491.91 | 0.00 | Trail-exit close>VWAP |

### Cycle 68 — BUY (started 2024-12-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 10:30:00 | 2511.00 | 2508.49 | 0.00 | ORB-long ORB[2480.00,2504.00] vol=2.6x ATR=6.69 |
| Stop hit — per-position SL triggered | 2024-12-27 10:40:00 | 2504.31 | 2507.82 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2024-12-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-30 11:05:00 | 2465.00 | 2474.22 | 0.00 | ORB-short ORB[2466.05,2490.00] vol=3.6x ATR=6.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 11:45:00 | 2456.00 | 2467.39 | 0.00 | T1 1.5R @ 2456.00 |
| Stop hit — per-position SL triggered | 2024-12-30 12:10:00 | 2465.00 | 2465.62 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-01-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 11:00:00 | 2450.40 | 2443.31 | 0.00 | ORB-long ORB[2432.60,2449.10] vol=3.9x ATR=6.26 |
| Stop hit — per-position SL triggered | 2025-01-01 11:25:00 | 2444.14 | 2444.08 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-01-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 10:50:00 | 2490.00 | 2471.29 | 0.00 | ORB-long ORB[2445.00,2467.00] vol=1.7x ATR=7.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 11:25:00 | 2501.14 | 2477.67 | 0.00 | T1 1.5R @ 2501.14 |
| Target hit | 2025-01-02 15:20:00 | 2542.30 | 2521.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 72 — SELL (started 2025-01-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 10:40:00 | 2520.15 | 2536.20 | 0.00 | ORB-short ORB[2535.05,2552.05] vol=1.5x ATR=6.35 |
| Stop hit — per-position SL triggered | 2025-01-03 10:55:00 | 2526.50 | 2532.20 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2025-01-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 10:45:00 | 2494.05 | 2511.14 | 0.00 | ORB-short ORB[2512.75,2525.95] vol=1.6x ATR=7.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:00:00 | 2483.04 | 2508.35 | 0.00 | T1 1.5R @ 2483.04 |
| Target hit | 2025-01-06 12:55:00 | 2490.00 | 2489.61 | 0.00 | Trail-exit close>VWAP |

### Cycle 74 — BUY (started 2025-01-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 10:45:00 | 2483.60 | 2481.87 | 0.00 | ORB-long ORB[2456.60,2481.60] vol=4.6x ATR=7.93 |
| Stop hit — per-position SL triggered | 2025-01-07 11:45:00 | 2475.67 | 2482.18 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2025-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 11:15:00 | 2449.50 | 2453.39 | 0.00 | ORB-short ORB[2451.30,2476.35] vol=1.6x ATR=5.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 11:20:00 | 2440.65 | 2452.87 | 0.00 | T1 1.5R @ 2440.65 |
| Target hit | 2025-01-08 14:00:00 | 2445.95 | 2445.89 | 0.00 | Trail-exit close>VWAP |

### Cycle 76 — SELL (started 2025-01-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 10:45:00 | 2401.05 | 2414.35 | 0.00 | ORB-short ORB[2418.00,2438.80] vol=4.1x ATR=6.66 |
| Stop hit — per-position SL triggered | 2025-01-09 11:00:00 | 2407.71 | 2413.20 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2025-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 11:15:00 | 2313.20 | 2323.67 | 0.00 | ORB-short ORB[2316.05,2348.20] vol=1.9x ATR=7.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 12:55:00 | 2301.49 | 2318.13 | 0.00 | T1 1.5R @ 2301.49 |
| Target hit | 2025-01-13 15:20:00 | 2288.15 | 2306.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 78 — SELL (started 2025-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-14 10:15:00 | 2295.55 | 2308.72 | 0.00 | ORB-short ORB[2301.35,2327.85] vol=2.8x ATR=7.80 |
| Stop hit — per-position SL triggered | 2025-01-14 10:35:00 | 2303.35 | 2303.57 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2025-01-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 10:00:00 | 2347.20 | 2334.85 | 0.00 | ORB-long ORB[2313.15,2346.70] vol=2.4x ATR=8.10 |
| Stop hit — per-position SL triggered | 2025-01-16 10:30:00 | 2339.10 | 2341.89 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2025-01-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 09:55:00 | 2393.00 | 2382.55 | 0.00 | ORB-long ORB[2375.85,2387.05] vol=1.5x ATR=6.74 |
| Stop hit — per-position SL triggered | 2025-01-20 10:05:00 | 2386.26 | 2384.43 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2025-01-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-22 11:00:00 | 2388.55 | 2384.95 | 0.00 | ORB-long ORB[2370.30,2387.95] vol=2.0x ATR=6.40 |
| Stop hit — per-position SL triggered | 2025-01-22 11:35:00 | 2382.15 | 2385.31 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2025-01-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 11:05:00 | 2436.25 | 2462.87 | 0.00 | ORB-short ORB[2467.10,2481.80] vol=1.8x ATR=7.58 |
| Stop hit — per-position SL triggered | 2025-01-27 11:20:00 | 2443.83 | 2460.41 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2025-01-31 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 10:25:00 | 2501.85 | 2494.52 | 0.00 | ORB-long ORB[2475.80,2500.60] vol=1.8x ATR=8.19 |
| Stop hit — per-position SL triggered | 2025-01-31 10:45:00 | 2493.66 | 2496.81 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2025-02-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-05 10:40:00 | 2479.65 | 2486.64 | 0.00 | ORB-short ORB[2482.00,2499.00] vol=2.1x ATR=6.63 |
| Stop hit — per-position SL triggered | 2025-02-05 11:25:00 | 2486.28 | 2485.17 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2025-02-14 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 10:35:00 | 2468.50 | 2482.47 | 0.00 | ORB-short ORB[2484.00,2507.75] vol=1.5x ATR=7.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 11:35:00 | 2457.47 | 2472.56 | 0.00 | T1 1.5R @ 2457.47 |
| Target hit | 2025-02-14 15:20:00 | 2429.70 | 2442.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 86 — SELL (started 2025-02-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-24 10:20:00 | 2393.00 | 2402.78 | 0.00 | ORB-short ORB[2402.80,2420.00] vol=2.0x ATR=5.62 |
| Stop hit — per-position SL triggered | 2025-02-24 10:30:00 | 2398.62 | 2399.88 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2025-02-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 10:35:00 | 2400.00 | 2390.30 | 0.00 | ORB-long ORB[2367.10,2399.90] vol=1.8x ATR=7.16 |
| Stop hit — per-position SL triggered | 2025-02-25 11:05:00 | 2392.84 | 2392.14 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2025-03-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-04 10:55:00 | 2378.50 | 2371.62 | 0.00 | ORB-long ORB[2351.60,2377.00] vol=1.6x ATR=6.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-04 13:25:00 | 2388.20 | 2377.46 | 0.00 | T1 1.5R @ 2388.20 |
| Target hit | 2025-03-04 15:20:00 | 2392.80 | 2384.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 89 — SELL (started 2025-03-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-05 11:05:00 | 2380.00 | 2391.20 | 0.00 | ORB-short ORB[2383.85,2408.95] vol=2.1x ATR=5.04 |
| Stop hit — per-position SL triggered | 2025-03-05 11:30:00 | 2385.04 | 2389.29 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2025-03-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 11:00:00 | 2398.45 | 2384.18 | 0.00 | ORB-long ORB[2363.95,2390.40] vol=3.6x ATR=6.87 |
| Stop hit — per-position SL triggered | 2025-03-11 11:15:00 | 2391.58 | 2385.13 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2025-03-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 11:00:00 | 2370.00 | 2401.98 | 0.00 | ORB-short ORB[2409.05,2438.90] vol=2.0x ATR=6.14 |
| Stop hit — per-position SL triggered | 2025-03-12 11:10:00 | 2376.14 | 2399.85 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2025-03-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:35:00 | 2420.70 | 2413.86 | 0.00 | ORB-long ORB[2397.00,2411.45] vol=1.9x ATR=5.06 |
| Stop hit — per-position SL triggered | 2025-03-18 14:55:00 | 2415.64 | 2418.81 | 0.00 | SL hit |

### Cycle 93 — BUY (started 2025-03-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 09:50:00 | 2472.55 | 2464.51 | 0.00 | ORB-long ORB[2448.10,2470.00] vol=2.3x ATR=5.77 |
| Stop hit — per-position SL triggered | 2025-03-20 10:05:00 | 2466.78 | 2467.87 | 0.00 | SL hit |

### Cycle 94 — BUY (started 2025-03-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-25 10:10:00 | 2549.00 | 2537.76 | 0.00 | ORB-long ORB[2520.00,2540.55] vol=4.8x ATR=7.93 |
| Stop hit — per-position SL triggered | 2025-03-25 10:15:00 | 2541.07 | 2538.36 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2025-03-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 09:30:00 | 2610.75 | 2597.06 | 0.00 | ORB-long ORB[2575.00,2601.00] vol=2.8x ATR=6.66 |
| Stop hit — per-position SL triggered | 2025-03-27 09:35:00 | 2604.09 | 2598.96 | 0.00 | SL hit |

### Cycle 96 — BUY (started 2025-04-03 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-03 10:25:00 | 2637.00 | 2621.01 | 0.00 | ORB-long ORB[2595.10,2619.00] vol=1.5x ATR=6.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-03 10:55:00 | 2646.87 | 2626.91 | 0.00 | T1 1.5R @ 2646.87 |
| Stop hit — per-position SL triggered | 2025-04-03 11:15:00 | 2637.00 | 2628.86 | 0.00 | SL hit |

### Cycle 97 — BUY (started 2025-04-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-08 10:05:00 | 2572.35 | 2566.26 | 0.00 | ORB-long ORB[2534.15,2565.55] vol=2.1x ATR=10.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-08 10:20:00 | 2588.64 | 2567.80 | 0.00 | T1 1.5R @ 2588.64 |
| Stop hit — per-position SL triggered | 2025-04-08 10:40:00 | 2572.35 | 2569.20 | 0.00 | SL hit |

### Cycle 98 — SELL (started 2025-04-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-09 11:00:00 | 2534.90 | 2566.41 | 0.00 | ORB-short ORB[2562.50,2586.75] vol=2.2x ATR=7.79 |
| Stop hit — per-position SL triggered | 2025-04-09 11:15:00 | 2542.69 | 2564.42 | 0.00 | SL hit |

### Cycle 99 — BUY (started 2025-04-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 09:55:00 | 2737.10 | 2715.36 | 0.00 | ORB-long ORB[2686.90,2710.60] vol=2.2x ATR=8.72 |
| Stop hit — per-position SL triggered | 2025-04-17 10:05:00 | 2728.38 | 2721.38 | 0.00 | SL hit |

### Cycle 100 — BUY (started 2025-04-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 11:10:00 | 2774.70 | 2762.99 | 0.00 | ORB-long ORB[2735.00,2763.00] vol=2.0x ATR=6.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 11:30:00 | 2783.78 | 2765.92 | 0.00 | T1 1.5R @ 2783.78 |
| Stop hit — per-position SL triggered | 2025-04-21 11:50:00 | 2774.70 | 2768.84 | 0.00 | SL hit |

### Cycle 101 — SELL (started 2025-04-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:35:00 | 2720.30 | 2735.66 | 0.00 | ORB-short ORB[2726.40,2763.90] vol=2.4x ATR=8.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 10:40:00 | 2707.16 | 2730.60 | 0.00 | T1 1.5R @ 2707.16 |
| Target hit | 2025-04-23 15:20:00 | 2686.30 | 2700.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 102 — BUY (started 2025-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 11:15:00 | 2723.20 | 2704.69 | 0.00 | ORB-long ORB[2673.00,2704.60] vol=1.7x ATR=6.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-24 11:55:00 | 2733.40 | 2710.27 | 0.00 | T1 1.5R @ 2733.40 |
| Stop hit — per-position SL triggered | 2025-04-24 13:30:00 | 2723.20 | 2716.41 | 0.00 | SL hit |

### Cycle 103 — SELL (started 2025-04-25 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 10:20:00 | 2706.90 | 2708.86 | 0.00 | ORB-short ORB[2710.00,2735.90] vol=5.8x ATR=7.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 10:25:00 | 2695.29 | 2708.41 | 0.00 | T1 1.5R @ 2695.29 |
| Target hit | 2025-04-25 11:55:00 | 2700.90 | 2698.75 | 0.00 | Trail-exit close>VWAP |

### Cycle 104 — BUY (started 2025-04-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-28 09:50:00 | 2768.10 | 2748.83 | 0.00 | ORB-long ORB[2723.30,2747.70] vol=1.8x ATR=8.98 |
| Stop hit — per-position SL triggered | 2025-04-28 10:30:00 | 2759.12 | 2754.66 | 0.00 | SL hit |

### Cycle 105 — SELL (started 2025-04-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 09:40:00 | 2728.10 | 2740.27 | 0.00 | ORB-short ORB[2740.50,2753.40] vol=3.8x ATR=6.48 |
| Stop hit — per-position SL triggered | 2025-04-29 09:55:00 | 2734.58 | 2739.94 | 0.00 | SL hit |

### Cycle 106 — BUY (started 2025-05-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 10:50:00 | 2756.00 | 2751.56 | 0.00 | ORB-long ORB[2718.30,2749.90] vol=1.7x ATR=7.70 |
| Stop hit — per-position SL triggered | 2025-05-05 11:50:00 | 2748.30 | 2752.16 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-13 10:55:00 | 2349.95 | 2024-05-13 11:15:00 | 2360.66 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-05-14 11:05:00 | 2370.10 | 2024-05-14 11:25:00 | 2362.60 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-05-14 11:05:00 | 2370.10 | 2024-05-14 11:35:00 | 2370.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-16 10:20:00 | 2351.60 | 2024-05-16 10:45:00 | 2341.13 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-05-16 10:20:00 | 2351.60 | 2024-05-16 12:10:00 | 2351.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-17 10:55:00 | 2417.75 | 2024-05-17 12:30:00 | 2409.82 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-05-22 10:55:00 | 2431.55 | 2024-05-22 11:05:00 | 2421.05 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-05-22 10:55:00 | 2431.55 | 2024-05-22 11:30:00 | 2431.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-27 10:35:00 | 2414.15 | 2024-05-27 10:50:00 | 2403.05 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-05-27 10:35:00 | 2414.15 | 2024-05-27 15:20:00 | 2390.35 | TARGET_HIT | 0.50 | 0.99% |
| BUY | retest1 | 2024-05-29 10:40:00 | 2441.55 | 2024-05-29 11:20:00 | 2452.19 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-05-29 10:40:00 | 2441.55 | 2024-05-29 11:30:00 | 2441.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-10 09:45:00 | 2405.05 | 2024-06-10 10:00:00 | 2416.90 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-06-10 09:45:00 | 2405.05 | 2024-06-10 15:10:00 | 2445.00 | TARGET_HIT | 0.50 | 1.66% |
| BUY | retest1 | 2024-06-12 09:30:00 | 2455.25 | 2024-06-12 09:45:00 | 2466.39 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-06-12 09:30:00 | 2455.25 | 2024-06-12 10:05:00 | 2455.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-21 10:50:00 | 2484.05 | 2024-06-21 12:25:00 | 2472.03 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-06-21 10:50:00 | 2484.05 | 2024-06-21 14:50:00 | 2484.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-26 09:30:00 | 2537.25 | 2024-06-26 09:55:00 | 2547.91 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-06-26 09:30:00 | 2537.25 | 2024-06-26 10:30:00 | 2537.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-02 10:20:00 | 2757.55 | 2024-07-02 12:05:00 | 2749.91 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-03 09:30:00 | 2753.80 | 2024-07-03 13:05:00 | 2745.91 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-07-04 10:30:00 | 2718.15 | 2024-07-04 10:35:00 | 2724.46 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-07-11 10:40:00 | 2833.00 | 2024-07-11 10:55:00 | 2825.07 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-12 10:45:00 | 2830.45 | 2024-07-12 10:55:00 | 2839.77 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-07-12 10:45:00 | 2830.45 | 2024-07-12 12:20:00 | 2830.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-16 10:05:00 | 2828.55 | 2024-07-16 10:20:00 | 2837.13 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-07-16 10:05:00 | 2828.55 | 2024-07-16 11:20:00 | 2828.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-18 11:15:00 | 2799.50 | 2024-07-18 11:20:00 | 2806.63 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-07-26 10:55:00 | 2856.80 | 2024-07-26 11:00:00 | 2849.49 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-07-29 10:50:00 | 2831.20 | 2024-07-29 11:00:00 | 2837.62 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-08-02 10:35:00 | 2734.80 | 2024-08-02 10:55:00 | 2742.29 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-08-05 11:10:00 | 2629.20 | 2024-08-05 11:20:00 | 2615.42 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-08-05 11:10:00 | 2629.20 | 2024-08-05 12:00:00 | 2629.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-06 10:20:00 | 2659.35 | 2024-08-06 10:30:00 | 2652.29 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-08-08 11:00:00 | 2597.55 | 2024-08-08 11:10:00 | 2586.30 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-08-08 11:00:00 | 2597.55 | 2024-08-08 11:35:00 | 2597.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-13 10:15:00 | 2535.35 | 2024-08-13 12:40:00 | 2522.60 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-08-13 10:15:00 | 2535.35 | 2024-08-13 13:10:00 | 2535.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-20 10:00:00 | 2620.35 | 2024-08-20 11:00:00 | 2613.08 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-08-21 10:45:00 | 2651.05 | 2024-08-21 14:00:00 | 2660.21 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-08-21 10:45:00 | 2651.05 | 2024-08-21 15:20:00 | 2691.65 | TARGET_HIT | 0.50 | 1.53% |
| SELL | retest1 | 2024-08-29 10:50:00 | 2674.85 | 2024-08-29 11:05:00 | 2664.93 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-08-29 10:50:00 | 2674.85 | 2024-08-29 11:25:00 | 2674.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-04 09:55:00 | 2724.65 | 2024-09-04 10:05:00 | 2736.22 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-09-04 09:55:00 | 2724.65 | 2024-09-04 10:40:00 | 2724.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-06 11:05:00 | 2696.55 | 2024-09-06 11:25:00 | 2703.73 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-09-10 10:00:00 | 2692.60 | 2024-09-10 10:25:00 | 2682.39 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-09-10 10:00:00 | 2692.60 | 2024-09-10 11:15:00 | 2692.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-11 10:50:00 | 2719.35 | 2024-09-11 11:15:00 | 2713.18 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-09-16 11:05:00 | 2788.30 | 2024-09-16 13:35:00 | 2777.20 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-09-16 11:05:00 | 2788.30 | 2024-09-16 15:20:00 | 2767.30 | TARGET_HIT | 0.50 | 0.75% |
| BUY | retest1 | 2024-09-18 09:55:00 | 2764.10 | 2024-09-18 10:10:00 | 2774.16 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-09-18 09:55:00 | 2764.10 | 2024-09-18 10:20:00 | 2764.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-20 10:25:00 | 2706.20 | 2024-09-20 10:45:00 | 2714.48 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-09-25 11:10:00 | 2596.00 | 2024-09-25 11:15:00 | 2602.77 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-09-27 09:45:00 | 2790.10 | 2024-09-27 10:20:00 | 2779.73 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-09-30 10:20:00 | 2816.25 | 2024-09-30 10:30:00 | 2805.63 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-10-03 09:50:00 | 2815.70 | 2024-10-03 10:05:00 | 2829.64 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-10-03 09:50:00 | 2815.70 | 2024-10-03 10:15:00 | 2815.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-07 10:35:00 | 2736.75 | 2024-10-07 11:00:00 | 2724.06 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-10-07 10:35:00 | 2736.75 | 2024-10-07 11:05:00 | 2736.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-08 11:15:00 | 2731.00 | 2024-10-08 11:35:00 | 2742.19 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-10-08 11:15:00 | 2731.00 | 2024-10-08 12:05:00 | 2731.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-10 10:50:00 | 2733.00 | 2024-10-10 11:05:00 | 2725.84 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-10-11 11:05:00 | 2693.60 | 2024-10-11 11:50:00 | 2699.58 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-10-14 09:45:00 | 2719.10 | 2024-10-14 10:55:00 | 2709.69 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-10-14 09:45:00 | 2719.10 | 2024-10-14 13:40:00 | 2716.40 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2024-10-15 09:30:00 | 2745.30 | 2024-10-15 09:40:00 | 2739.06 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-10-16 10:05:00 | 2756.95 | 2024-10-16 10:10:00 | 2767.67 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-10-16 10:05:00 | 2756.95 | 2024-10-16 10:15:00 | 2756.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-17 09:35:00 | 2762.00 | 2024-10-17 09:45:00 | 2749.71 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-10-17 09:35:00 | 2762.00 | 2024-10-17 10:25:00 | 2762.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-18 10:25:00 | 2725.00 | 2024-10-18 12:05:00 | 2740.08 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-10-18 10:25:00 | 2725.00 | 2024-10-18 15:20:00 | 2761.80 | TARGET_HIT | 0.50 | 1.35% |
| SELL | retest1 | 2024-10-22 10:40:00 | 2693.75 | 2024-10-22 10:45:00 | 2703.58 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-10-23 09:55:00 | 2633.35 | 2024-10-23 10:05:00 | 2645.88 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-10-24 09:45:00 | 2670.20 | 2024-10-24 10:05:00 | 2686.82 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-10-24 09:45:00 | 2670.20 | 2024-10-24 10:45:00 | 2670.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-25 10:50:00 | 2623.45 | 2024-10-25 10:55:00 | 2631.43 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-10-28 09:40:00 | 2595.80 | 2024-10-28 09:45:00 | 2605.14 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-10-29 10:00:00 | 2676.00 | 2024-10-29 10:05:00 | 2667.61 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-10-30 11:05:00 | 2699.65 | 2024-10-30 11:15:00 | 2693.65 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-11-04 09:55:00 | 2664.60 | 2024-11-04 10:10:00 | 2648.34 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-11-04 09:55:00 | 2664.60 | 2024-11-04 15:20:00 | 2592.75 | TARGET_HIT | 0.50 | 2.70% |
| SELL | retest1 | 2024-11-06 11:10:00 | 2622.30 | 2024-11-06 11:30:00 | 2629.68 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-11-13 09:45:00 | 2494.05 | 2024-11-13 09:50:00 | 2502.92 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-11-28 10:30:00 | 2600.35 | 2024-11-28 10:40:00 | 2589.72 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-11-28 10:30:00 | 2600.35 | 2024-11-28 10:45:00 | 2600.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-29 11:00:00 | 2607.25 | 2024-11-29 11:15:00 | 2598.77 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-12-02 11:00:00 | 2677.85 | 2024-12-02 11:30:00 | 2690.53 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-12-02 11:00:00 | 2677.85 | 2024-12-02 15:20:00 | 2691.20 | TARGET_HIT | 0.50 | 0.50% |
| SELL | retest1 | 2024-12-05 10:55:00 | 2680.35 | 2024-12-05 12:05:00 | 2688.07 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-12-09 10:15:00 | 2671.85 | 2024-12-09 10:30:00 | 2677.84 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-12 11:10:00 | 2652.75 | 2024-12-12 11:35:00 | 2657.82 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-12-18 11:10:00 | 2590.00 | 2024-12-18 11:25:00 | 2596.27 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-12-23 11:00:00 | 2544.80 | 2024-12-23 11:55:00 | 2537.79 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-12-26 10:30:00 | 2495.40 | 2024-12-26 12:30:00 | 2484.82 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-12-26 10:30:00 | 2495.40 | 2024-12-26 14:35:00 | 2492.20 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2024-12-27 10:30:00 | 2511.00 | 2024-12-27 10:40:00 | 2504.31 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-12-30 11:05:00 | 2465.00 | 2024-12-30 11:45:00 | 2456.00 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-12-30 11:05:00 | 2465.00 | 2024-12-30 12:10:00 | 2465.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-01 11:00:00 | 2450.40 | 2025-01-01 11:25:00 | 2444.14 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-01-02 10:50:00 | 2490.00 | 2025-01-02 11:25:00 | 2501.14 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-01-02 10:50:00 | 2490.00 | 2025-01-02 15:20:00 | 2542.30 | TARGET_HIT | 0.50 | 2.10% |
| SELL | retest1 | 2025-01-03 10:40:00 | 2520.15 | 2025-01-03 10:55:00 | 2526.50 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-01-06 10:45:00 | 2494.05 | 2025-01-06 11:00:00 | 2483.04 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-01-06 10:45:00 | 2494.05 | 2025-01-06 12:55:00 | 2490.00 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2025-01-07 10:45:00 | 2483.60 | 2025-01-07 11:45:00 | 2475.67 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-01-08 11:15:00 | 2449.50 | 2025-01-08 11:20:00 | 2440.65 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-01-08 11:15:00 | 2449.50 | 2025-01-08 14:00:00 | 2445.95 | TARGET_HIT | 0.50 | 0.14% |
| SELL | retest1 | 2025-01-09 10:45:00 | 2401.05 | 2025-01-09 11:00:00 | 2407.71 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-01-13 11:15:00 | 2313.20 | 2025-01-13 12:55:00 | 2301.49 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-01-13 11:15:00 | 2313.20 | 2025-01-13 15:20:00 | 2288.15 | TARGET_HIT | 0.50 | 1.08% |
| SELL | retest1 | 2025-01-14 10:15:00 | 2295.55 | 2025-01-14 10:35:00 | 2303.35 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-01-16 10:00:00 | 2347.20 | 2025-01-16 10:30:00 | 2339.10 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-01-20 09:55:00 | 2393.00 | 2025-01-20 10:05:00 | 2386.26 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-01-22 11:00:00 | 2388.55 | 2025-01-22 11:35:00 | 2382.15 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-01-27 11:05:00 | 2436.25 | 2025-01-27 11:20:00 | 2443.83 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-01-31 10:25:00 | 2501.85 | 2025-01-31 10:45:00 | 2493.66 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-02-05 10:40:00 | 2479.65 | 2025-02-05 11:25:00 | 2486.28 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-02-14 10:35:00 | 2468.50 | 2025-02-14 11:35:00 | 2457.47 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-02-14 10:35:00 | 2468.50 | 2025-02-14 15:20:00 | 2429.70 | TARGET_HIT | 0.50 | 1.57% |
| SELL | retest1 | 2025-02-24 10:20:00 | 2393.00 | 2025-02-24 10:30:00 | 2398.62 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-02-25 10:35:00 | 2400.00 | 2025-02-25 11:05:00 | 2392.84 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-03-04 10:55:00 | 2378.50 | 2025-03-04 13:25:00 | 2388.20 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-03-04 10:55:00 | 2378.50 | 2025-03-04 15:20:00 | 2392.80 | TARGET_HIT | 0.50 | 0.60% |
| SELL | retest1 | 2025-03-05 11:05:00 | 2380.00 | 2025-03-05 11:30:00 | 2385.04 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-03-11 11:00:00 | 2398.45 | 2025-03-11 11:15:00 | 2391.58 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-03-12 11:00:00 | 2370.00 | 2025-03-12 11:10:00 | 2376.14 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-03-18 10:35:00 | 2420.70 | 2025-03-18 14:55:00 | 2415.64 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-03-20 09:50:00 | 2472.55 | 2025-03-20 10:05:00 | 2466.78 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-03-25 10:10:00 | 2549.00 | 2025-03-25 10:15:00 | 2541.07 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-03-27 09:30:00 | 2610.75 | 2025-03-27 09:35:00 | 2604.09 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-04-03 10:25:00 | 2637.00 | 2025-04-03 10:55:00 | 2646.87 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-04-03 10:25:00 | 2637.00 | 2025-04-03 11:15:00 | 2637.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-08 10:05:00 | 2572.35 | 2025-04-08 10:20:00 | 2588.64 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-04-08 10:05:00 | 2572.35 | 2025-04-08 10:40:00 | 2572.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-09 11:00:00 | 2534.90 | 2025-04-09 11:15:00 | 2542.69 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-04-17 09:55:00 | 2737.10 | 2025-04-17 10:05:00 | 2728.38 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-04-21 11:10:00 | 2774.70 | 2025-04-21 11:30:00 | 2783.78 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-04-21 11:10:00 | 2774.70 | 2025-04-21 11:50:00 | 2774.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-23 10:35:00 | 2720.30 | 2025-04-23 10:40:00 | 2707.16 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-04-23 10:35:00 | 2720.30 | 2025-04-23 15:20:00 | 2686.30 | TARGET_HIT | 0.50 | 1.25% |
| BUY | retest1 | 2025-04-24 11:15:00 | 2723.20 | 2025-04-24 11:55:00 | 2733.40 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-04-24 11:15:00 | 2723.20 | 2025-04-24 13:30:00 | 2723.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-25 10:20:00 | 2706.90 | 2025-04-25 10:25:00 | 2695.29 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-04-25 10:20:00 | 2706.90 | 2025-04-25 11:55:00 | 2700.90 | TARGET_HIT | 0.50 | 0.22% |
| BUY | retest1 | 2025-04-28 09:50:00 | 2768.10 | 2025-04-28 10:30:00 | 2759.12 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-04-29 09:40:00 | 2728.10 | 2025-04-29 09:55:00 | 2734.58 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-05-05 10:50:00 | 2756.00 | 2025-05-05 11:50:00 | 2748.30 | STOP_HIT | 1.00 | -0.28% |
