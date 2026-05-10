# Glaxosmithkline Pharmaceuticals Ltd. (GLAXO)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 2480.40
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
| ENTRY1 | 28 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 35 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 25
- **Target hits / Stop hits / Partials:** 3 / 25 / 7
- **Avg / median % per leg:** -0.06% / -0.28%
- **Sum % (uncompounded):** -2.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 6 | 23.1% | 2 | 20 | 4 | -0.11% | -2.9% |
| BUY @ 2nd Alert (retest1) | 26 | 6 | 23.1% | 2 | 20 | 4 | -0.11% | -2.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 4 | 44.4% | 1 | 5 | 3 | 0.08% | 0.7% |
| SELL @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 1 | 5 | 3 | 0.08% | 0.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 35 | 10 | 28.6% | 3 | 25 | 7 | -0.06% | -2.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:25:00 | 2537.00 | 2509.60 | 0.00 | ORB-long ORB[2475.00,2509.90] vol=1.9x ATR=13.73 |
| Stop hit — per-position SL triggered | 2026-02-09 10:45:00 | 2523.27 | 2511.09 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 09:55:00 | 2582.40 | 2586.74 | 0.00 | ORB-short ORB[2586.90,2608.00] vol=5.0x ATR=7.17 |
| Stop hit — per-position SL triggered | 2026-02-17 10:00:00 | 2589.57 | 2588.11 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 2610.00 | 2604.46 | 0.00 | ORB-long ORB[2584.00,2609.90] vol=2.3x ATR=5.95 |
| Stop hit — per-position SL triggered | 2026-02-19 09:35:00 | 2604.05 | 2605.01 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:45:00 | 2586.40 | 2575.54 | 0.00 | ORB-long ORB[2558.10,2584.80] vol=3.3x ATR=7.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:50:00 | 2597.35 | 2580.74 | 0.00 | T1 1.5R @ 2597.35 |
| Stop hit — per-position SL triggered | 2026-02-20 15:00:00 | 2586.40 | 2589.94 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 10:35:00 | 2647.50 | 2633.58 | 0.00 | ORB-long ORB[2610.50,2638.70] vol=1.6x ATR=8.16 |
| Stop hit — per-position SL triggered | 2026-02-23 10:40:00 | 2639.34 | 2634.24 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:30:00 | 2610.00 | 2619.45 | 0.00 | ORB-short ORB[2612.00,2635.10] vol=3.6x ATR=6.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 10:40:00 | 2600.11 | 2615.41 | 0.00 | T1 1.5R @ 2600.11 |
| Stop hit — per-position SL triggered | 2026-02-24 12:20:00 | 2610.00 | 2609.51 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 11:00:00 | 2642.60 | 2635.14 | 0.00 | ORB-long ORB[2602.50,2635.20] vol=6.2x ATR=5.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:15:00 | 2650.62 | 2636.54 | 0.00 | T1 1.5R @ 2650.62 |
| Stop hit — per-position SL triggered | 2026-02-25 13:25:00 | 2642.60 | 2646.97 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-02-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:50:00 | 2583.90 | 2597.77 | 0.00 | ORB-short ORB[2595.40,2633.10] vol=6.2x ATR=6.29 |
| Stop hit — per-position SL triggered | 2026-02-27 12:30:00 | 2590.19 | 2594.61 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-02 11:05:00 | 2552.80 | 2549.74 | 0.00 | ORB-long ORB[2511.70,2545.00] vol=5.9x ATR=10.99 |
| Stop hit — per-position SL triggered | 2026-03-02 11:30:00 | 2541.81 | 2549.65 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 09:45:00 | 2515.50 | 2510.99 | 0.00 | ORB-long ORB[2487.00,2512.00] vol=1.6x ATR=7.72 |
| Stop hit — per-position SL triggered | 2026-03-05 09:55:00 | 2507.78 | 2510.95 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:45:00 | 2505.80 | 2499.34 | 0.00 | ORB-long ORB[2482.00,2499.00] vol=2.6x ATR=7.51 |
| Stop hit — per-position SL triggered | 2026-03-11 10:00:00 | 2498.29 | 2499.53 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 09:40:00 | 2427.20 | 2411.23 | 0.00 | ORB-long ORB[2390.20,2420.00] vol=2.5x ATR=10.92 |
| Stop hit — per-position SL triggered | 2026-03-16 09:55:00 | 2416.28 | 2412.98 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-03-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:20:00 | 2418.00 | 2411.93 | 0.00 | ORB-long ORB[2396.00,2417.30] vol=1.8x ATR=7.14 |
| Stop hit — per-position SL triggered | 2026-03-17 10:25:00 | 2410.86 | 2411.94 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-03-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 10:35:00 | 2463.40 | 2450.39 | 0.00 | ORB-long ORB[2418.20,2453.90] vol=2.9x ATR=7.03 |
| Stop hit — per-position SL triggered | 2026-03-18 11:55:00 | 2456.37 | 2456.04 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-03-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-19 10:05:00 | 2414.30 | 2402.39 | 0.00 | ORB-long ORB[2392.10,2409.30] vol=1.5x ATR=8.38 |
| Stop hit — per-position SL triggered | 2026-03-19 11:30:00 | 2405.92 | 2407.00 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-03-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:35:00 | 2408.00 | 2403.17 | 0.00 | ORB-long ORB[2376.80,2405.90] vol=1.5x ATR=7.20 |
| Stop hit — per-position SL triggered | 2026-03-20 09:40:00 | 2400.80 | 2403.41 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-03-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 11:05:00 | 2319.80 | 2337.28 | 0.00 | ORB-short ORB[2340.60,2364.90] vol=2.0x ATR=9.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 12:20:00 | 2306.29 | 2329.21 | 0.00 | T1 1.5R @ 2306.29 |
| Target hit | 2026-03-23 14:05:00 | 2316.40 | 2316.06 | 0.00 | Trail-exit close>VWAP |

### Cycle 18 — BUY (started 2026-04-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 09:30:00 | 2379.50 | 2361.83 | 0.00 | ORB-long ORB[2341.20,2370.10] vol=1.5x ATR=8.95 |
| Stop hit — per-position SL triggered | 2026-04-13 09:35:00 | 2370.55 | 2366.06 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2026-04-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:35:00 | 2416.60 | 2418.19 | 0.00 | ORB-short ORB[2417.10,2441.60] vol=1.8x ATR=6.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 10:55:00 | 2406.81 | 2417.88 | 0.00 | T1 1.5R @ 2406.81 |
| Stop hit — per-position SL triggered | 2026-04-16 12:45:00 | 2416.60 | 2412.60 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2026-04-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:35:00 | 2435.00 | 2426.19 | 0.00 | ORB-long ORB[2410.00,2428.00] vol=2.1x ATR=6.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:00:00 | 2444.78 | 2432.74 | 0.00 | T1 1.5R @ 2444.78 |
| Target hit | 2026-04-17 15:20:00 | 2452.50 | 2449.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — BUY (started 2026-04-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:45:00 | 2442.20 | 2436.15 | 0.00 | ORB-long ORB[2420.10,2437.60] vol=2.2x ATR=6.98 |
| Stop hit — per-position SL triggered | 2026-04-21 11:00:00 | 2435.22 | 2436.52 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2026-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:30:00 | 2437.70 | 2427.62 | 0.00 | ORB-long ORB[2416.30,2436.90] vol=1.8x ATR=6.33 |
| Stop hit — per-position SL triggered | 2026-04-22 09:50:00 | 2431.37 | 2430.90 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2026-04-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:40:00 | 2513.90 | 2489.96 | 0.00 | ORB-long ORB[2461.90,2480.00] vol=2.5x ATR=11.58 |
| Stop hit — per-position SL triggered | 2026-04-23 10:20:00 | 2502.32 | 2507.84 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2026-04-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:45:00 | 2495.00 | 2476.15 | 0.00 | ORB-long ORB[2457.40,2486.10] vol=6.5x ATR=10.12 |
| Stop hit — per-position SL triggered | 2026-04-27 09:50:00 | 2484.88 | 2475.67 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2026-04-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 11:00:00 | 2383.50 | 2408.23 | 0.00 | ORB-short ORB[2403.40,2428.50] vol=2.3x ATR=7.24 |
| Stop hit — per-position SL triggered | 2026-04-29 11:20:00 | 2390.74 | 2405.08 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2026-05-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:55:00 | 2400.00 | 2389.35 | 0.00 | ORB-long ORB[2365.00,2392.00] vol=3.4x ATR=8.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:10:00 | 2412.92 | 2390.83 | 0.00 | T1 1.5R @ 2412.92 |
| Target hit | 2026-05-06 15:20:00 | 2424.10 | 2416.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2026-05-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:35:00 | 2461.10 | 2453.57 | 0.00 | ORB-long ORB[2423.50,2459.90] vol=2.6x ATR=9.83 |
| Stop hit — per-position SL triggered | 2026-05-07 09:50:00 | 2451.27 | 2454.07 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2026-05-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:30:00 | 2484.90 | 2466.66 | 0.00 | ORB-long ORB[2458.50,2484.00] vol=2.9x ATR=7.65 |
| Stop hit — per-position SL triggered | 2026-05-08 10:40:00 | 2477.25 | 2472.28 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:25:00 | 2537.00 | 2026-02-09 10:45:00 | 2523.27 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2026-02-17 09:55:00 | 2582.40 | 2026-02-17 10:00:00 | 2589.57 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-19 09:30:00 | 2610.00 | 2026-02-19 09:35:00 | 2604.05 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-02-20 10:45:00 | 2586.40 | 2026-02-20 11:50:00 | 2597.35 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-02-20 10:45:00 | 2586.40 | 2026-02-20 15:00:00 | 2586.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-23 10:35:00 | 2647.50 | 2026-02-23 10:40:00 | 2639.34 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-24 10:30:00 | 2610.00 | 2026-02-24 10:40:00 | 2600.11 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-02-24 10:30:00 | 2610.00 | 2026-02-24 12:20:00 | 2610.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 11:00:00 | 2642.60 | 2026-02-25 11:15:00 | 2650.62 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-02-25 11:00:00 | 2642.60 | 2026-02-25 13:25:00 | 2642.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 10:50:00 | 2583.90 | 2026-02-27 12:30:00 | 2590.19 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-03-02 11:05:00 | 2552.80 | 2026-03-02 11:30:00 | 2541.81 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-03-05 09:45:00 | 2515.50 | 2026-03-05 09:55:00 | 2507.78 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-03-11 09:45:00 | 2505.80 | 2026-03-11 10:00:00 | 2498.29 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-03-16 09:40:00 | 2427.20 | 2026-03-16 09:55:00 | 2416.28 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-03-17 10:20:00 | 2418.00 | 2026-03-17 10:25:00 | 2410.86 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-03-18 10:35:00 | 2463.40 | 2026-03-18 11:55:00 | 2456.37 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-03-19 10:05:00 | 2414.30 | 2026-03-19 11:30:00 | 2405.92 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-03-20 09:35:00 | 2408.00 | 2026-03-20 09:40:00 | 2400.80 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-23 11:05:00 | 2319.80 | 2026-03-23 12:20:00 | 2306.29 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-03-23 11:05:00 | 2319.80 | 2026-03-23 14:05:00 | 2316.40 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2026-04-13 09:30:00 | 2379.50 | 2026-04-13 09:35:00 | 2370.55 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-04-16 10:35:00 | 2416.60 | 2026-04-16 10:55:00 | 2406.81 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-04-16 10:35:00 | 2416.60 | 2026-04-16 12:45:00 | 2416.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 09:35:00 | 2435.00 | 2026-04-17 10:00:00 | 2444.78 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-04-17 09:35:00 | 2435.00 | 2026-04-17 15:20:00 | 2452.50 | TARGET_HIT | 0.50 | 0.72% |
| BUY | retest1 | 2026-04-21 10:45:00 | 2442.20 | 2026-04-21 11:00:00 | 2435.22 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-22 09:30:00 | 2437.70 | 2026-04-22 09:50:00 | 2431.37 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-04-23 09:40:00 | 2513.90 | 2026-04-23 10:20:00 | 2502.32 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2026-04-27 09:45:00 | 2495.00 | 2026-04-27 09:50:00 | 2484.88 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-04-29 11:00:00 | 2383.50 | 2026-04-29 11:20:00 | 2390.74 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-06 09:55:00 | 2400.00 | 2026-05-06 10:10:00 | 2412.92 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-05-06 09:55:00 | 2400.00 | 2026-05-06 15:20:00 | 2424.10 | TARGET_HIT | 0.50 | 1.00% |
| BUY | retest1 | 2026-05-07 09:35:00 | 2461.10 | 2026-05-07 09:50:00 | 2451.27 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-05-08 10:30:00 | 2484.90 | 2026-05-08 10:40:00 | 2477.25 | STOP_HIT | 1.00 | -0.31% |
