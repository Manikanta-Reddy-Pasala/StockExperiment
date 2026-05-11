# ASIANPAINT (ASIANPAINT)

## Backtest Summary

- **Window:** 2025-06-10 09:15:00 → 2026-05-08 15:25:00 (13963 bars)
- **Last close:** 2600.00
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
| ENTRY1 | 57 |
| ENTRY2 | 0 |
| PARTIAL | 25 |
| TARGET_HIT | 9 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 82 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 34 / 48
- **Target hits / Stop hits / Partials:** 9 / 48 / 25
- **Avg / median % per leg:** 0.07% / 0.00%
- **Sum % (uncompounded):** 5.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 43 | 17 | 39.5% | 5 | 26 | 12 | 0.05% | 2.1% |
| BUY @ 2nd Alert (retest1) | 43 | 17 | 39.5% | 5 | 26 | 12 | 0.05% | 2.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 39 | 17 | 43.6% | 4 | 22 | 13 | 0.10% | 3.8% |
| SELL @ 2nd Alert (retest1) | 39 | 17 | 43.6% | 4 | 22 | 13 | 0.10% | 3.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 82 | 34 | 41.5% | 9 | 48 | 25 | 0.07% | 5.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:30:00 | 2261.80 | 2242.21 | 0.00 | ORB-long ORB[2227.00,2254.00] vol=2.0x ATR=5.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-17 09:35:00 | 2270.16 | 2248.44 | 0.00 | T1 1.5R @ 2270.16 |
| Target hit | 2025-06-17 12:40:00 | 2262.60 | 2264.23 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — BUY (started 2025-06-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 10:05:00 | 2317.60 | 2301.78 | 0.00 | ORB-long ORB[2277.10,2300.00] vol=2.4x ATR=5.37 |
| Stop hit — per-position SL triggered | 2025-06-27 10:20:00 | 2312.23 | 2304.73 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-07-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 09:30:00 | 2394.30 | 2385.39 | 0.00 | ORB-long ORB[2365.00,2392.70] vol=3.1x ATR=7.32 |
| Stop hit — per-position SL triggered | 2025-07-01 09:35:00 | 2386.98 | 2385.22 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-07-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 09:35:00 | 2452.90 | 2441.92 | 0.00 | ORB-long ORB[2421.00,2449.00] vol=1.7x ATR=5.92 |
| Stop hit — per-position SL triggered | 2025-07-07 09:40:00 | 2446.98 | 2443.45 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-07-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 09:35:00 | 2530.50 | 2517.15 | 0.00 | ORB-long ORB[2495.70,2524.90] vol=2.6x ATR=7.81 |
| Stop hit — per-position SL triggered | 2025-07-09 09:45:00 | 2522.69 | 2518.96 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-07-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 09:55:00 | 2475.30 | 2491.28 | 0.00 | ORB-short ORB[2489.40,2512.00] vol=2.0x ATR=6.98 |
| Stop hit — per-position SL triggered | 2025-07-10 11:55:00 | 2482.28 | 2483.77 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-07-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-11 09:30:00 | 2466.70 | 2458.51 | 0.00 | ORB-long ORB[2442.10,2464.00] vol=2.8x ATR=5.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 09:50:00 | 2474.56 | 2462.40 | 0.00 | T1 1.5R @ 2474.56 |
| Stop hit — per-position SL triggered | 2025-07-11 11:05:00 | 2466.70 | 2467.11 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-07-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-15 11:05:00 | 2396.40 | 2399.08 | 0.00 | ORB-short ORB[2406.00,2415.00] vol=2.4x ATR=3.83 |
| Stop hit — per-position SL triggered | 2025-07-15 11:10:00 | 2400.23 | 2399.11 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-07-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 09:50:00 | 2374.10 | 2383.29 | 0.00 | ORB-short ORB[2380.60,2393.40] vol=1.6x ATR=4.32 |
| Stop hit — per-position SL triggered | 2025-07-16 10:50:00 | 2378.42 | 2378.54 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-07-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:10:00 | 2390.40 | 2394.53 | 0.00 | ORB-short ORB[2393.30,2405.00] vol=1.6x ATR=4.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:15:00 | 2384.30 | 2394.12 | 0.00 | T1 1.5R @ 2384.30 |
| Target hit | 2025-07-18 14:45:00 | 2386.00 | 2384.98 | 0.00 | Trail-exit close>VWAP |

### Cycle 11 — SELL (started 2025-07-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 11:00:00 | 2367.20 | 2368.17 | 0.00 | ORB-short ORB[2367.80,2380.00] vol=2.2x ATR=3.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 11:25:00 | 2361.98 | 2367.44 | 0.00 | T1 1.5R @ 2361.98 |
| Target hit | 2025-07-24 15:20:00 | 2350.30 | 2356.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2025-07-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 10:50:00 | 2355.00 | 2341.88 | 0.00 | ORB-long ORB[2320.00,2343.00] vol=1.5x ATR=5.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 11:25:00 | 2362.73 | 2345.60 | 0.00 | T1 1.5R @ 2362.73 |
| Stop hit — per-position SL triggered | 2025-07-28 12:40:00 | 2355.00 | 2349.61 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-06 10:15:00 | 2464.10 | 2453.12 | 0.00 | ORB-long ORB[2440.00,2453.20] vol=1.5x ATR=5.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 10:25:00 | 2472.11 | 2458.44 | 0.00 | T1 1.5R @ 2472.11 |
| Target hit | 2025-08-06 15:20:00 | 2491.00 | 2486.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 11:15:00 | 2474.00 | 2484.27 | 0.00 | ORB-short ORB[2475.70,2491.20] vol=2.5x ATR=4.95 |
| Stop hit — per-position SL triggered | 2025-08-07 11:50:00 | 2478.95 | 2482.42 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-08-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-08 11:05:00 | 2494.30 | 2495.76 | 0.00 | ORB-short ORB[2495.00,2511.70] vol=3.3x ATR=5.10 |
| Stop hit — per-position SL triggered | 2025-08-08 11:15:00 | 2499.40 | 2495.86 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-10-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 10:55:00 | 2338.50 | 2352.29 | 0.00 | ORB-short ORB[2354.00,2369.40] vol=1.9x ATR=7.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 12:30:00 | 2327.99 | 2347.34 | 0.00 | T1 1.5R @ 2327.99 |
| Target hit | 2025-10-08 15:20:00 | 2326.20 | 2336.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — SELL (started 2025-10-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 11:00:00 | 2327.30 | 2339.91 | 0.00 | ORB-short ORB[2341.50,2355.00] vol=2.2x ATR=3.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 11:30:00 | 2321.34 | 2337.04 | 0.00 | T1 1.5R @ 2321.34 |
| Stop hit — per-position SL triggered | 2025-10-14 14:35:00 | 2327.30 | 2330.54 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-10-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 09:30:00 | 2351.50 | 2342.71 | 0.00 | ORB-long ORB[2325.00,2346.80] vol=3.6x ATR=5.80 |
| Stop hit — per-position SL triggered | 2025-10-15 09:35:00 | 2345.70 | 2343.58 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-10-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 10:35:00 | 2520.00 | 2515.72 | 0.00 | ORB-long ORB[2499.60,2518.60] vol=2.3x ATR=3.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 10:40:00 | 2525.94 | 2517.51 | 0.00 | T1 1.5R @ 2525.94 |
| Stop hit — per-position SL triggered | 2025-10-27 10:50:00 | 2520.00 | 2517.86 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-10-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 09:40:00 | 2527.60 | 2517.08 | 0.00 | ORB-long ORB[2506.20,2521.70] vol=2.5x ATR=5.03 |
| Stop hit — per-position SL triggered | 2025-10-28 09:45:00 | 2522.57 | 2517.40 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-11-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 09:30:00 | 2902.20 | 2891.61 | 0.00 | ORB-long ORB[2862.00,2899.40] vol=1.6x ATR=7.53 |
| Stop hit — per-position SL triggered | 2025-11-14 09:45:00 | 2894.67 | 2893.19 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-11-21 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:25:00 | 2860.00 | 2863.80 | 0.00 | ORB-short ORB[2862.20,2876.00] vol=1.9x ATR=4.15 |
| Stop hit — per-position SL triggered | 2025-11-21 10:40:00 | 2864.15 | 2863.36 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-11-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 09:40:00 | 2905.70 | 2896.37 | 0.00 | ORB-long ORB[2877.50,2901.60] vol=2.8x ATR=6.64 |
| Stop hit — per-position SL triggered | 2025-11-27 10:10:00 | 2899.06 | 2901.32 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-12-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 09:30:00 | 2971.00 | 2962.18 | 0.00 | ORB-long ORB[2934.00,2968.00] vol=1.9x ATR=6.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 09:45:00 | 2980.61 | 2967.59 | 0.00 | T1 1.5R @ 2980.61 |
| Target hit | 2025-12-04 10:55:00 | 2972.00 | 2973.68 | 0.00 | Trail-exit close<VWAP |

### Cycle 25 — SELL (started 2025-12-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:50:00 | 2940.90 | 2948.70 | 0.00 | ORB-short ORB[2948.10,2974.50] vol=1.7x ATR=4.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 11:05:00 | 2934.11 | 2945.52 | 0.00 | T1 1.5R @ 2934.11 |
| Stop hit — per-position SL triggered | 2025-12-08 12:15:00 | 2940.90 | 2942.79 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-12-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-12 11:00:00 | 2756.90 | 2778.80 | 0.00 | ORB-short ORB[2778.70,2797.00] vol=2.1x ATR=6.53 |
| Stop hit — per-position SL triggered | 2025-12-12 11:20:00 | 2763.43 | 2777.40 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-12-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 10:55:00 | 2796.30 | 2784.98 | 0.00 | ORB-long ORB[2758.70,2789.80] vol=1.7x ATR=5.70 |
| Stop hit — per-position SL triggered | 2025-12-15 11:35:00 | 2790.60 | 2786.53 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-12-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 10:00:00 | 2786.60 | 2774.67 | 0.00 | ORB-long ORB[2759.70,2776.90] vol=1.8x ATR=5.06 |
| Stop hit — per-position SL triggered | 2025-12-19 10:10:00 | 2781.54 | 2777.91 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-12-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 10:45:00 | 2796.30 | 2808.56 | 0.00 | ORB-short ORB[2803.50,2819.60] vol=10.7x ATR=6.05 |
| Stop hit — per-position SL triggered | 2025-12-24 11:25:00 | 2802.35 | 2805.84 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-12-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 10:55:00 | 2767.50 | 2770.49 | 0.00 | ORB-short ORB[2771.80,2794.40] vol=3.9x ATR=4.72 |
| Stop hit — per-position SL triggered | 2025-12-26 11:00:00 | 2772.22 | 2770.44 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-12-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 11:00:00 | 2743.20 | 2753.03 | 0.00 | ORB-short ORB[2746.50,2760.90] vol=2.4x ATR=5.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 11:05:00 | 2735.46 | 2752.23 | 0.00 | T1 1.5R @ 2735.46 |
| Stop hit — per-position SL triggered | 2025-12-29 11:25:00 | 2743.20 | 2750.44 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-12-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 10:55:00 | 2781.90 | 2774.70 | 0.00 | ORB-long ORB[2760.00,2779.10] vol=3.8x ATR=5.43 |
| Stop hit — per-position SL triggered | 2025-12-30 11:10:00 | 2776.47 | 2775.64 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2026-01-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 10:45:00 | 2781.60 | 2774.29 | 0.00 | ORB-long ORB[2761.40,2778.70] vol=1.8x ATR=4.79 |
| Stop hit — per-position SL triggered | 2026-01-01 10:50:00 | 2776.81 | 2774.52 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2026-01-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 09:35:00 | 2800.60 | 2793.10 | 0.00 | ORB-long ORB[2769.00,2797.00] vol=1.9x ATR=6.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 09:45:00 | 2810.00 | 2800.97 | 0.00 | T1 1.5R @ 2810.00 |
| Stop hit — per-position SL triggered | 2026-01-05 10:00:00 | 2800.60 | 2802.63 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2026-01-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 10:45:00 | 2884.60 | 2893.28 | 0.00 | ORB-short ORB[2891.00,2914.50] vol=4.2x ATR=6.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-13 10:55:00 | 2875.18 | 2891.39 | 0.00 | T1 1.5R @ 2875.18 |
| Stop hit — per-position SL triggered | 2026-01-13 11:25:00 | 2884.60 | 2888.86 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 10:15:00 | 2821.10 | 2807.57 | 0.00 | ORB-long ORB[2780.10,2820.50] vol=2.2x ATR=7.37 |
| Stop hit — per-position SL triggered | 2026-01-16 11:10:00 | 2813.73 | 2810.81 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2026-01-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-19 10:25:00 | 2769.90 | 2763.79 | 0.00 | ORB-long ORB[2750.30,2768.90] vol=3.9x ATR=6.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 10:55:00 | 2779.34 | 2767.71 | 0.00 | T1 1.5R @ 2779.34 |
| Stop hit — per-position SL triggered | 2026-01-19 12:35:00 | 2769.90 | 2771.90 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2026-01-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-23 10:55:00 | 2774.70 | 2742.76 | 0.00 | ORB-long ORB[2690.80,2730.00] vol=2.3x ATR=7.35 |
| Stop hit — per-position SL triggered | 2026-01-23 11:15:00 | 2767.35 | 2750.95 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2026-02-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-02 09:35:00 | 2411.80 | 2394.12 | 0.00 | ORB-long ORB[2372.00,2406.90] vol=2.7x ATR=14.64 |
| Stop hit — per-position SL triggered | 2026-02-02 11:00:00 | 2397.16 | 2405.36 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 2405.10 | 2408.23 | 0.00 | ORB-short ORB[2407.30,2420.00] vol=6.1x ATR=5.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:35:00 | 2397.41 | 2406.95 | 0.00 | T1 1.5R @ 2397.41 |
| Stop hit — per-position SL triggered | 2026-02-10 09:45:00 | 2405.10 | 2406.50 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2026-02-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 10:10:00 | 2383.90 | 2389.25 | 0.00 | ORB-short ORB[2393.60,2405.70] vol=4.1x ATR=4.80 |
| Stop hit — per-position SL triggered | 2026-02-11 10:35:00 | 2388.70 | 2389.11 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 11:15:00 | 2396.00 | 2400.37 | 0.00 | ORB-short ORB[2400.30,2411.80] vol=1.8x ATR=4.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 11:35:00 | 2389.28 | 2399.09 | 0.00 | T1 1.5R @ 2389.28 |
| Target hit | 2026-02-13 15:20:00 | 2365.20 | 2381.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 11:15:00 | 2379.30 | 2368.74 | 0.00 | ORB-long ORB[2356.80,2372.60] vol=1.6x ATR=4.74 |
| Stop hit — per-position SL triggered | 2026-02-16 11:30:00 | 2374.56 | 2369.36 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2026-02-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:00:00 | 2432.90 | 2434.14 | 0.00 | ORB-short ORB[2433.50,2448.20] vol=1.5x ATR=4.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:20:00 | 2426.70 | 2433.62 | 0.00 | T1 1.5R @ 2426.70 |
| Stop hit — per-position SL triggered | 2026-02-18 12:55:00 | 2432.90 | 2428.90 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2026-02-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:10:00 | 2400.00 | 2409.80 | 0.00 | ORB-short ORB[2401.00,2428.00] vol=1.8x ATR=5.05 |
| Stop hit — per-position SL triggered | 2026-02-19 11:25:00 | 2405.05 | 2409.57 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2026-02-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:55:00 | 2404.00 | 2412.70 | 0.00 | ORB-short ORB[2410.00,2420.30] vol=1.8x ATR=3.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:05:00 | 2398.32 | 2410.72 | 0.00 | T1 1.5R @ 2398.32 |
| Stop hit — per-position SL triggered | 2026-02-26 11:25:00 | 2404.00 | 2408.74 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2026-02-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:40:00 | 2369.70 | 2375.86 | 0.00 | ORB-short ORB[2371.40,2397.60] vol=1.5x ATR=4.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:00:00 | 2362.26 | 2371.64 | 0.00 | T1 1.5R @ 2362.26 |
| Stop hit — per-position SL triggered | 2026-02-27 11:25:00 | 2369.70 | 2371.01 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2026-03-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:25:00 | 2257.00 | 2269.22 | 0.00 | ORB-short ORB[2271.40,2302.10] vol=1.8x ATR=5.84 |
| Stop hit — per-position SL triggered | 2026-03-05 10:35:00 | 2262.84 | 2268.77 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2026-03-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:55:00 | 2271.50 | 2286.77 | 0.00 | ORB-short ORB[2285.30,2302.00] vol=2.8x ATR=6.35 |
| Stop hit — per-position SL triggered | 2026-03-11 11:00:00 | 2277.85 | 2286.03 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2026-03-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:20:00 | 2234.60 | 2219.65 | 0.00 | ORB-long ORB[2201.00,2223.60] vol=1.5x ATR=7.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 10:45:00 | 2245.39 | 2223.95 | 0.00 | T1 1.5R @ 2245.39 |
| Stop hit — per-position SL triggered | 2026-03-12 11:50:00 | 2234.60 | 2230.28 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2026-03-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 10:50:00 | 2220.20 | 2213.30 | 0.00 | ORB-long ORB[2204.10,2219.60] vol=2.2x ATR=4.77 |
| Stop hit — per-position SL triggered | 2026-03-20 11:00:00 | 2215.43 | 2213.53 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2026-03-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:10:00 | 2248.40 | 2241.31 | 0.00 | ORB-long ORB[2222.00,2242.90] vol=1.9x ATR=6.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 10:40:00 | 2258.18 | 2244.92 | 0.00 | T1 1.5R @ 2258.18 |
| Target hit | 2026-03-25 15:20:00 | 2270.00 | 2264.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — BUY (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 11:15:00 | 2473.50 | 2455.27 | 0.00 | ORB-long ORB[2417.50,2446.90] vol=1.7x ATR=5.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 11:50:00 | 2482.15 | 2458.63 | 0.00 | T1 1.5R @ 2482.15 |
| Stop hit — per-position SL triggered | 2026-04-17 12:50:00 | 2473.50 | 2463.45 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2026-04-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:40:00 | 2544.40 | 2537.14 | 0.00 | ORB-long ORB[2513.70,2537.90] vol=4.2x ATR=7.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:45:00 | 2555.15 | 2539.40 | 0.00 | T1 1.5R @ 2555.15 |
| Target hit | 2026-04-21 12:05:00 | 2558.80 | 2560.48 | 0.00 | Trail-exit close<VWAP |

### Cycle 55 — BUY (started 2026-04-28 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:25:00 | 2494.00 | 2483.79 | 0.00 | ORB-long ORB[2458.00,2492.10] vol=2.4x ATR=6.81 |
| Stop hit — per-position SL triggered | 2026-04-28 11:00:00 | 2487.19 | 2486.20 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2026-04-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 11:10:00 | 2397.20 | 2413.14 | 0.00 | ORB-short ORB[2408.70,2429.70] vol=2.2x ATR=6.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 11:20:00 | 2387.02 | 2409.71 | 0.00 | T1 1.5R @ 2387.02 |
| Stop hit — per-position SL triggered | 2026-04-30 11:35:00 | 2397.20 | 2407.72 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2026-05-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:40:00 | 2481.30 | 2453.11 | 0.00 | ORB-long ORB[2440.40,2469.80] vol=2.3x ATR=6.95 |
| Stop hit — per-position SL triggered | 2026-05-06 10:45:00 | 2474.35 | 2454.17 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-17 09:30:00 | 2261.80 | 2025-06-17 09:35:00 | 2270.16 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-06-17 09:30:00 | 2261.80 | 2025-06-17 12:40:00 | 2262.60 | TARGET_HIT | 0.50 | 0.04% |
| BUY | retest1 | 2025-06-27 10:05:00 | 2317.60 | 2025-06-27 10:20:00 | 2312.23 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-07-01 09:30:00 | 2394.30 | 2025-07-01 09:35:00 | 2386.98 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-07-07 09:35:00 | 2452.90 | 2025-07-07 09:40:00 | 2446.98 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-07-09 09:35:00 | 2530.50 | 2025-07-09 09:45:00 | 2522.69 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-07-10 09:55:00 | 2475.30 | 2025-07-10 11:55:00 | 2482.28 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-07-11 09:30:00 | 2466.70 | 2025-07-11 09:50:00 | 2474.56 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-07-11 09:30:00 | 2466.70 | 2025-07-11 11:05:00 | 2466.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-15 11:05:00 | 2396.40 | 2025-07-15 11:10:00 | 2400.23 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-07-16 09:50:00 | 2374.10 | 2025-07-16 10:50:00 | 2378.42 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-07-18 10:10:00 | 2390.40 | 2025-07-18 10:15:00 | 2384.30 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-07-18 10:10:00 | 2390.40 | 2025-07-18 14:45:00 | 2386.00 | TARGET_HIT | 0.50 | 0.18% |
| SELL | retest1 | 2025-07-24 11:00:00 | 2367.20 | 2025-07-24 11:25:00 | 2361.98 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2025-07-24 11:00:00 | 2367.20 | 2025-07-24 15:20:00 | 2350.30 | TARGET_HIT | 0.50 | 0.71% |
| BUY | retest1 | 2025-07-28 10:50:00 | 2355.00 | 2025-07-28 11:25:00 | 2362.73 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-07-28 10:50:00 | 2355.00 | 2025-07-28 12:40:00 | 2355.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-06 10:15:00 | 2464.10 | 2025-08-06 10:25:00 | 2472.11 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-08-06 10:15:00 | 2464.10 | 2025-08-06 15:20:00 | 2491.00 | TARGET_HIT | 0.50 | 1.09% |
| SELL | retest1 | 2025-08-07 11:15:00 | 2474.00 | 2025-08-07 11:50:00 | 2478.95 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-08-08 11:05:00 | 2494.30 | 2025-08-08 11:15:00 | 2499.40 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-10-08 10:55:00 | 2338.50 | 2025-10-08 12:30:00 | 2327.99 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-10-08 10:55:00 | 2338.50 | 2025-10-08 15:20:00 | 2326.20 | TARGET_HIT | 0.50 | 0.53% |
| SELL | retest1 | 2025-10-14 11:00:00 | 2327.30 | 2025-10-14 11:30:00 | 2321.34 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-10-14 11:00:00 | 2327.30 | 2025-10-14 14:35:00 | 2327.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-15 09:30:00 | 2351.50 | 2025-10-15 09:35:00 | 2345.70 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-10-27 10:35:00 | 2520.00 | 2025-10-27 10:40:00 | 2525.94 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2025-10-27 10:35:00 | 2520.00 | 2025-10-27 10:50:00 | 2520.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-28 09:40:00 | 2527.60 | 2025-10-28 09:45:00 | 2522.57 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-11-14 09:30:00 | 2902.20 | 2025-11-14 09:45:00 | 2894.67 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-11-21 10:25:00 | 2860.00 | 2025-11-21 10:40:00 | 2864.15 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-11-27 09:40:00 | 2905.70 | 2025-11-27 10:10:00 | 2899.06 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-12-04 09:30:00 | 2971.00 | 2025-12-04 09:45:00 | 2980.61 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-12-04 09:30:00 | 2971.00 | 2025-12-04 10:55:00 | 2972.00 | TARGET_HIT | 0.50 | 0.03% |
| SELL | retest1 | 2025-12-08 10:50:00 | 2940.90 | 2025-12-08 11:05:00 | 2934.11 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2025-12-08 10:50:00 | 2940.90 | 2025-12-08 12:15:00 | 2940.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-12 11:00:00 | 2756.90 | 2025-12-12 11:20:00 | 2763.43 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-12-15 10:55:00 | 2796.30 | 2025-12-15 11:35:00 | 2790.60 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-12-19 10:00:00 | 2786.60 | 2025-12-19 10:10:00 | 2781.54 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-12-24 10:45:00 | 2796.30 | 2025-12-24 11:25:00 | 2802.35 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-12-26 10:55:00 | 2767.50 | 2025-12-26 11:00:00 | 2772.22 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-12-29 11:00:00 | 2743.20 | 2025-12-29 11:05:00 | 2735.46 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-12-29 11:00:00 | 2743.20 | 2025-12-29 11:25:00 | 2743.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-30 10:55:00 | 2781.90 | 2025-12-30 11:10:00 | 2776.47 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-01-01 10:45:00 | 2781.60 | 2026-01-01 10:50:00 | 2776.81 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2026-01-05 09:35:00 | 2800.60 | 2026-01-05 09:45:00 | 2810.00 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2026-01-05 09:35:00 | 2800.60 | 2026-01-05 10:00:00 | 2800.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-13 10:45:00 | 2884.60 | 2026-01-13 10:55:00 | 2875.18 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-01-13 10:45:00 | 2884.60 | 2026-01-13 11:25:00 | 2884.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-16 10:15:00 | 2821.10 | 2026-01-16 11:10:00 | 2813.73 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-01-19 10:25:00 | 2769.90 | 2026-01-19 10:55:00 | 2779.34 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2026-01-19 10:25:00 | 2769.90 | 2026-01-19 12:35:00 | 2769.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-23 10:55:00 | 2774.70 | 2026-01-23 11:15:00 | 2767.35 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-02 09:35:00 | 2411.80 | 2026-02-02 11:00:00 | 2397.16 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest1 | 2026-02-10 09:30:00 | 2405.10 | 2026-02-10 09:35:00 | 2397.41 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-02-10 09:30:00 | 2405.10 | 2026-02-10 09:45:00 | 2405.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-11 10:10:00 | 2383.90 | 2026-02-11 10:35:00 | 2388.70 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-13 11:15:00 | 2396.00 | 2026-02-13 11:35:00 | 2389.28 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2026-02-13 11:15:00 | 2396.00 | 2026-02-13 15:20:00 | 2365.20 | TARGET_HIT | 0.50 | 1.29% |
| BUY | retest1 | 2026-02-16 11:15:00 | 2379.30 | 2026-02-16 11:30:00 | 2374.56 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-18 11:00:00 | 2432.90 | 2026-02-18 11:20:00 | 2426.70 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2026-02-18 11:00:00 | 2432.90 | 2026-02-18 12:55:00 | 2432.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 11:10:00 | 2400.00 | 2026-02-19 11:25:00 | 2405.05 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-26 10:55:00 | 2404.00 | 2026-02-26 11:05:00 | 2398.32 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2026-02-26 10:55:00 | 2404.00 | 2026-02-26 11:25:00 | 2404.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 09:40:00 | 2369.70 | 2026-02-27 11:00:00 | 2362.26 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-02-27 09:40:00 | 2369.70 | 2026-02-27 11:25:00 | 2369.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-05 10:25:00 | 2257.00 | 2026-03-05 10:35:00 | 2262.84 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-03-11 10:55:00 | 2271.50 | 2026-03-11 11:00:00 | 2277.85 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-03-12 10:20:00 | 2234.60 | 2026-03-12 10:45:00 | 2245.39 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-03-12 10:20:00 | 2234.60 | 2026-03-12 11:50:00 | 2234.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-20 10:50:00 | 2220.20 | 2026-03-20 11:00:00 | 2215.43 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-03-25 10:10:00 | 2248.40 | 2026-03-25 10:40:00 | 2258.18 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-03-25 10:10:00 | 2248.40 | 2026-03-25 15:20:00 | 2270.00 | TARGET_HIT | 0.50 | 0.96% |
| BUY | retest1 | 2026-04-17 11:15:00 | 2473.50 | 2026-04-17 11:50:00 | 2482.15 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-04-17 11:15:00 | 2473.50 | 2026-04-17 12:50:00 | 2473.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 09:40:00 | 2544.40 | 2026-04-21 09:45:00 | 2555.15 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-04-21 09:40:00 | 2544.40 | 2026-04-21 12:05:00 | 2558.80 | TARGET_HIT | 0.50 | 0.57% |
| BUY | retest1 | 2026-04-28 10:25:00 | 2494.00 | 2026-04-28 11:00:00 | 2487.19 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-04-30 11:10:00 | 2397.20 | 2026-04-30 11:20:00 | 2387.02 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-04-30 11:10:00 | 2397.20 | 2026-04-30 11:35:00 | 2397.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-06 10:40:00 | 2481.30 | 2026-05-06 10:45:00 | 2474.35 | STOP_HIT | 1.00 | -0.28% |
