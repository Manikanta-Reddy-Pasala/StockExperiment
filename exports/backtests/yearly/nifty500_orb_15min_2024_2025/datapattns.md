# Data Patterns (India) Ltd. (DATAPATTNS)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 4118.00
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
| ENTRY1 | 29 |
| ENTRY2 | 0 |
| PARTIAL | 11 |
| TARGET_HIT | 5 |
| STOP_HIT | 24 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 40 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 24
- **Target hits / Stop hits / Partials:** 5 / 24 / 11
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 6.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 6 | 33.3% | 2 | 12 | 4 | 0.03% | 0.5% |
| BUY @ 2nd Alert (retest1) | 18 | 6 | 33.3% | 2 | 12 | 4 | 0.03% | 0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 22 | 10 | 45.5% | 3 | 12 | 7 | 0.25% | 5.5% |
| SELL @ 2nd Alert (retest1) | 22 | 10 | 45.5% | 3 | 12 | 7 | 0.25% | 5.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 40 | 16 | 40.0% | 5 | 24 | 11 | 0.15% | 6.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 10:00:00 | 2929.00 | 2905.91 | 0.00 | ORB-long ORB[2876.55,2905.00] vol=1.6x ATR=14.70 |
| Stop hit — per-position SL triggered | 2024-05-16 10:20:00 | 2914.30 | 2913.09 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-31 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 09:50:00 | 2802.60 | 2850.64 | 0.00 | ORB-short ORB[2849.90,2888.85] vol=2.5x ATR=18.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 10:15:00 | 2774.64 | 2824.38 | 0.00 | T1 1.5R @ 2774.64 |
| Stop hit — per-position SL triggered | 2024-05-31 10:35:00 | 2802.60 | 2797.38 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 11:15:00 | 2713.80 | 2742.01 | 0.00 | ORB-short ORB[2735.05,2766.95] vol=2.0x ATR=7.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 11:25:00 | 2701.88 | 2740.49 | 0.00 | T1 1.5R @ 2701.88 |
| Stop hit — per-position SL triggered | 2024-06-13 11:50:00 | 2713.80 | 2738.00 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-06-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:30:00 | 2791.00 | 2768.74 | 0.00 | ORB-long ORB[2741.00,2780.00] vol=2.5x ATR=16.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 09:35:00 | 2815.52 | 2783.22 | 0.00 | T1 1.5R @ 2815.52 |
| Stop hit — per-position SL triggered | 2024-06-14 10:00:00 | 2791.00 | 2793.79 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:30:00 | 3005.95 | 2992.51 | 0.00 | ORB-long ORB[2948.35,2987.90] vol=4.6x ATR=18.59 |
| Stop hit — per-position SL triggered | 2024-06-21 09:35:00 | 2987.36 | 2992.61 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-07-05 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 10:30:00 | 3477.20 | 3354.84 | 0.00 | ORB-long ORB[3148.05,3199.30] vol=3.6x ATR=36.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 10:45:00 | 3532.11 | 3421.01 | 0.00 | T1 1.5R @ 3532.11 |
| Stop hit — per-position SL triggered | 2024-07-05 11:25:00 | 3477.20 | 3480.13 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-07-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-16 09:35:00 | 3327.90 | 3347.19 | 0.00 | ORB-short ORB[3337.40,3365.05] vol=1.9x ATR=16.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 13:55:00 | 3303.72 | 3330.04 | 0.00 | T1 1.5R @ 3303.72 |
| Target hit | 2024-07-16 15:20:00 | 3303.00 | 3317.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2024-07-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-31 09:30:00 | 3243.70 | 3267.67 | 0.00 | ORB-short ORB[3257.00,3303.20] vol=2.4x ATR=11.32 |
| Stop hit — per-position SL triggered | 2024-07-31 10:25:00 | 3255.02 | 3256.60 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-08-01 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 10:25:00 | 3185.15 | 3201.05 | 0.00 | ORB-short ORB[3186.00,3228.15] vol=1.5x ATR=12.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 10:45:00 | 3165.94 | 3198.31 | 0.00 | T1 1.5R @ 3165.94 |
| Stop hit — per-position SL triggered | 2024-08-01 11:05:00 | 3185.15 | 3197.18 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-08-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 10:40:00 | 3024.70 | 2993.68 | 0.00 | ORB-long ORB[2956.65,2995.80] vol=1.7x ATR=14.75 |
| Stop hit — per-position SL triggered | 2024-08-08 10:50:00 | 3009.95 | 2995.54 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-08-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:40:00 | 2986.70 | 2959.37 | 0.00 | ORB-long ORB[2937.50,2972.40] vol=1.9x ATR=15.29 |
| Stop hit — per-position SL triggered | 2024-08-16 09:45:00 | 2971.41 | 2960.82 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-22 10:15:00 | 2877.05 | 2895.33 | 0.00 | ORB-short ORB[2885.00,2911.00] vol=1.6x ATR=9.85 |
| Stop hit — per-position SL triggered | 2024-08-22 10:30:00 | 2886.90 | 2893.94 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-08-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:20:00 | 2825.65 | 2841.35 | 0.00 | ORB-short ORB[2826.00,2868.30] vol=3.1x ATR=8.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 10:45:00 | 2813.62 | 2837.23 | 0.00 | T1 1.5R @ 2813.62 |
| Target hit | 2024-08-29 15:20:00 | 2776.55 | 2802.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2024-09-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 09:45:00 | 2778.00 | 2803.78 | 0.00 | ORB-short ORB[2784.25,2817.00] vol=1.8x ATR=8.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:55:00 | 2764.76 | 2800.98 | 0.00 | T1 1.5R @ 2764.76 |
| Target hit | 2024-09-06 15:20:00 | 2738.15 | 2763.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — SELL (started 2024-09-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-11 10:55:00 | 2693.90 | 2707.14 | 0.00 | ORB-short ORB[2701.00,2732.00] vol=2.1x ATR=7.46 |
| Stop hit — per-position SL triggered | 2024-09-11 11:55:00 | 2701.36 | 2704.83 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-12 10:15:00 | 2677.60 | 2691.45 | 0.00 | ORB-short ORB[2685.00,2723.95] vol=1.8x ATR=6.87 |
| Stop hit — per-position SL triggered | 2024-09-12 10:20:00 | 2684.47 | 2691.25 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-09-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 09:55:00 | 2664.80 | 2678.08 | 0.00 | ORB-short ORB[2670.00,2696.80] vol=1.8x ATR=7.61 |
| Stop hit — per-position SL triggered | 2024-09-16 12:25:00 | 2672.41 | 2670.24 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-09-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 09:40:00 | 2708.10 | 2692.80 | 0.00 | ORB-long ORB[2679.30,2699.40] vol=2.1x ATR=9.22 |
| Stop hit — per-position SL triggered | 2024-09-18 11:15:00 | 2698.88 | 2698.30 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-09-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 09:45:00 | 2472.00 | 2481.57 | 0.00 | ORB-short ORB[2483.25,2518.80] vol=2.3x ATR=9.30 |
| Stop hit — per-position SL triggered | 2024-09-26 09:50:00 | 2481.30 | 2481.23 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-11-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 09:30:00 | 2165.15 | 2177.49 | 0.00 | ORB-short ORB[2170.00,2198.95] vol=1.6x ATR=9.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 09:40:00 | 2151.01 | 2172.09 | 0.00 | T1 1.5R @ 2151.01 |
| Stop hit — per-position SL triggered | 2024-11-18 10:00:00 | 2165.15 | 2167.95 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-11-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 09:55:00 | 2421.30 | 2398.18 | 0.00 | ORB-long ORB[2352.25,2388.35] vol=3.1x ATR=12.25 |
| Stop hit — per-position SL triggered | 2024-11-27 10:05:00 | 2409.05 | 2400.94 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-12-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-03 10:55:00 | 2525.00 | 2541.22 | 0.00 | ORB-short ORB[2530.00,2564.00] vol=1.5x ATR=8.20 |
| Stop hit — per-position SL triggered | 2024-12-03 11:10:00 | 2533.20 | 2540.35 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-12-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 10:10:00 | 2683.85 | 2663.94 | 0.00 | ORB-long ORB[2651.60,2682.00] vol=1.5x ATR=12.75 |
| Stop hit — per-position SL triggered | 2024-12-05 10:15:00 | 2671.10 | 2664.71 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 10:15:00 | 2662.35 | 2642.27 | 0.00 | ORB-long ORB[2620.85,2650.95] vol=1.6x ATR=12.21 |
| Stop hit — per-position SL triggered | 2024-12-12 10:20:00 | 2650.14 | 2642.66 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-12-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 10:00:00 | 2514.90 | 2495.10 | 0.00 | ORB-long ORB[2471.00,2502.50] vol=1.6x ATR=12.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 10:10:00 | 2532.93 | 2504.69 | 0.00 | T1 1.5R @ 2532.93 |
| Target hit | 2024-12-20 10:45:00 | 2534.50 | 2534.91 | 0.00 | Trail-exit close<VWAP |

### Cycle 26 — BUY (started 2025-01-01 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 09:55:00 | 2498.05 | 2483.71 | 0.00 | ORB-long ORB[2463.60,2494.00] vol=2.0x ATR=10.93 |
| Stop hit — per-position SL triggered | 2025-01-01 10:10:00 | 2487.12 | 2485.10 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-01-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 09:40:00 | 2530.35 | 2518.86 | 0.00 | ORB-long ORB[2490.00,2523.00] vol=4.1x ATR=8.47 |
| Stop hit — per-position SL triggered | 2025-01-03 09:50:00 | 2521.88 | 2519.98 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-01-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:40:00 | 2131.50 | 2120.63 | 0.00 | ORB-long ORB[2105.00,2130.05] vol=1.6x ATR=9.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 09:55:00 | 2146.49 | 2130.03 | 0.00 | T1 1.5R @ 2146.49 |
| Target hit | 2025-01-30 10:55:00 | 2143.05 | 2144.17 | 0.00 | Trail-exit close<VWAP |

### Cycle 29 — SELL (started 2025-04-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-17 09:35:00 | 1867.70 | 1883.89 | 0.00 | ORB-short ORB[1872.00,1899.80] vol=2.1x ATR=9.21 |
| Stop hit — per-position SL triggered | 2025-04-17 09:45:00 | 1876.91 | 1882.22 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-16 10:00:00 | 2929.00 | 2024-05-16 10:20:00 | 2914.30 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-05-31 09:50:00 | 2802.60 | 2024-05-31 10:15:00 | 2774.64 | PARTIAL | 0.50 | 1.00% |
| SELL | retest1 | 2024-05-31 09:50:00 | 2802.60 | 2024-05-31 10:35:00 | 2802.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-13 11:15:00 | 2713.80 | 2024-06-13 11:25:00 | 2701.88 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-06-13 11:15:00 | 2713.80 | 2024-06-13 11:50:00 | 2713.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-14 09:30:00 | 2791.00 | 2024-06-14 09:35:00 | 2815.52 | PARTIAL | 0.50 | 0.88% |
| BUY | retest1 | 2024-06-14 09:30:00 | 2791.00 | 2024-06-14 10:00:00 | 2791.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-21 09:30:00 | 3005.95 | 2024-06-21 09:35:00 | 2987.36 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest1 | 2024-07-05 10:30:00 | 3477.20 | 2024-07-05 10:45:00 | 3532.11 | PARTIAL | 0.50 | 1.58% |
| BUY | retest1 | 2024-07-05 10:30:00 | 3477.20 | 2024-07-05 11:25:00 | 3477.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-16 09:35:00 | 3327.90 | 2024-07-16 13:55:00 | 3303.72 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2024-07-16 09:35:00 | 3327.90 | 2024-07-16 15:20:00 | 3303.00 | TARGET_HIT | 0.50 | 0.75% |
| SELL | retest1 | 2024-07-31 09:30:00 | 3243.70 | 2024-07-31 10:25:00 | 3255.02 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-08-01 10:25:00 | 3185.15 | 2024-08-01 10:45:00 | 3165.94 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-08-01 10:25:00 | 3185.15 | 2024-08-01 11:05:00 | 3185.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-08 10:40:00 | 3024.70 | 2024-08-08 10:50:00 | 3009.95 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-08-16 09:40:00 | 2986.70 | 2024-08-16 09:45:00 | 2971.41 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2024-08-22 10:15:00 | 2877.05 | 2024-08-22 10:30:00 | 2886.90 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-08-29 10:20:00 | 2825.65 | 2024-08-29 10:45:00 | 2813.62 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-08-29 10:20:00 | 2825.65 | 2024-08-29 15:20:00 | 2776.55 | TARGET_HIT | 0.50 | 1.74% |
| SELL | retest1 | 2024-09-06 09:45:00 | 2778.00 | 2024-09-06 09:55:00 | 2764.76 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-09-06 09:45:00 | 2778.00 | 2024-09-06 15:20:00 | 2738.15 | TARGET_HIT | 0.50 | 1.43% |
| SELL | retest1 | 2024-09-11 10:55:00 | 2693.90 | 2024-09-11 11:55:00 | 2701.36 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-09-12 10:15:00 | 2677.60 | 2024-09-12 10:20:00 | 2684.47 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-09-16 09:55:00 | 2664.80 | 2024-09-16 12:25:00 | 2672.41 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-09-18 09:40:00 | 2708.10 | 2024-09-18 11:15:00 | 2698.88 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-09-26 09:45:00 | 2472.00 | 2024-09-26 09:50:00 | 2481.30 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-11-18 09:30:00 | 2165.15 | 2024-11-18 09:40:00 | 2151.01 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2024-11-18 09:30:00 | 2165.15 | 2024-11-18 10:00:00 | 2165.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-27 09:55:00 | 2421.30 | 2024-11-27 10:05:00 | 2409.05 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2024-12-03 10:55:00 | 2525.00 | 2024-12-03 11:10:00 | 2533.20 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-12-05 10:10:00 | 2683.85 | 2024-12-05 10:15:00 | 2671.10 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-12-12 10:15:00 | 2662.35 | 2024-12-12 10:20:00 | 2650.14 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-12-20 10:00:00 | 2514.90 | 2024-12-20 10:10:00 | 2532.93 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2024-12-20 10:00:00 | 2514.90 | 2024-12-20 10:45:00 | 2534.50 | TARGET_HIT | 0.50 | 0.78% |
| BUY | retest1 | 2025-01-01 09:55:00 | 2498.05 | 2025-01-01 10:10:00 | 2487.12 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-01-03 09:40:00 | 2530.35 | 2025-01-03 09:50:00 | 2521.88 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-01-30 09:40:00 | 2131.50 | 2025-01-30 09:55:00 | 2146.49 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2025-01-30 09:40:00 | 2131.50 | 2025-01-30 10:55:00 | 2143.05 | TARGET_HIT | 0.50 | 0.54% |
| SELL | retest1 | 2025-04-17 09:35:00 | 1867.70 | 2025-04-17 09:45:00 | 1876.91 | STOP_HIT | 1.00 | -0.49% |
