# Tata Consultancy Services Ltd. (TCS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 2397.20
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
| ENTRY1 | 17 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 14
- **Target hits / Stop hits / Partials:** 3 / 14 / 6
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 2.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 3 | 27.3% | 1 | 8 | 2 | -0.05% | -0.6% |
| BUY @ 2nd Alert (retest1) | 11 | 3 | 27.3% | 1 | 8 | 2 | -0.05% | -0.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 6 | 50.0% | 2 | 6 | 4 | 0.25% | 3.0% |
| SELL @ 2nd Alert (retest1) | 12 | 6 | 50.0% | 2 | 6 | 4 | 0.25% | 3.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 23 | 9 | 39.1% | 3 | 14 | 6 | 0.11% | 2.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 2969.70 | 2960.27 | 0.00 | ORB-long ORB[2943.50,2965.50] vol=1.7x ATR=5.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:40:00 | 2978.15 | 2964.62 | 0.00 | T1 1.5R @ 2978.15 |
| Stop hit — per-position SL triggered | 2026-02-10 09:55:00 | 2969.70 | 2965.93 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:35:00 | 2682.50 | 2687.61 | 0.00 | ORB-short ORB[2692.60,2730.00] vol=3.2x ATR=9.65 |
| Stop hit — per-position SL triggered | 2026-02-18 12:35:00 | 2692.15 | 2685.93 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 10:05:00 | 2745.40 | 2726.81 | 0.00 | ORB-long ORB[2706.10,2733.60] vol=1.9x ATR=8.34 |
| Stop hit — per-position SL triggered | 2026-02-19 10:15:00 | 2737.06 | 2728.76 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:35:00 | 2480.90 | 2495.86 | 0.00 | ORB-short ORB[2495.10,2521.00] vol=1.5x ATR=5.57 |
| Stop hit — per-position SL triggered | 2026-03-11 11:00:00 | 2486.47 | 2493.90 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 10:40:00 | 2441.00 | 2427.08 | 0.00 | ORB-long ORB[2419.00,2438.00] vol=1.5x ATR=6.34 |
| Stop hit — per-position SL triggered | 2026-03-13 10:45:00 | 2434.66 | 2427.57 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:20:00 | 2373.90 | 2398.72 | 0.00 | ORB-short ORB[2397.10,2421.60] vol=1.7x ATR=7.17 |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 2381.07 | 2390.87 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:25:00 | 2415.00 | 2405.19 | 0.00 | ORB-long ORB[2390.70,2408.20] vol=1.6x ATR=6.45 |
| Stop hit — per-position SL triggered | 2026-03-25 12:55:00 | 2408.55 | 2411.73 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-30 09:30:00 | 2392.30 | 2378.58 | 0.00 | ORB-long ORB[2355.00,2389.80] vol=1.8x ATR=7.44 |
| Stop hit — per-position SL triggered | 2026-03-30 09:50:00 | 2384.86 | 2382.92 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-01 11:00:00 | 2445.50 | 2445.07 | 0.00 | ORB-long ORB[2408.30,2443.30] vol=1.6x ATR=8.29 |
| Stop hit — per-position SL triggered | 2026-04-01 12:25:00 | 2437.21 | 2445.52 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-02 09:40:00 | 2412.60 | 2399.47 | 0.00 | ORB-long ORB[2375.70,2411.00] vol=2.0x ATR=9.01 |
| Stop hit — per-position SL triggered | 2026-04-02 09:45:00 | 2403.59 | 2400.51 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 09:35:00 | 2516.80 | 2539.15 | 0.00 | ORB-short ORB[2530.10,2565.80] vol=2.5x ATR=12.21 |
| Stop hit — per-position SL triggered | 2026-04-10 13:40:00 | 2529.01 | 2522.44 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:55:00 | 2587.30 | 2580.41 | 0.00 | ORB-long ORB[2560.00,2578.40] vol=1.5x ATR=5.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 11:35:00 | 2594.96 | 2582.96 | 0.00 | T1 1.5R @ 2594.96 |
| Target hit | 2026-04-21 15:20:00 | 2611.60 | 2596.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2026-04-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 11:10:00 | 2554.60 | 2567.09 | 0.00 | ORB-short ORB[2558.10,2580.00] vol=1.6x ATR=5.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 11:40:00 | 2546.98 | 2564.01 | 0.00 | T1 1.5R @ 2546.98 |
| Target hit | 2026-04-22 13:55:00 | 2550.20 | 2543.43 | 0.00 | Trail-exit close>VWAP |

### Cycle 14 — SELL (started 2026-04-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:45:00 | 2466.90 | 2481.45 | 0.00 | ORB-short ORB[2472.30,2505.00] vol=1.9x ATR=7.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:10:00 | 2455.51 | 2475.79 | 0.00 | T1 1.5R @ 2455.51 |
| Target hit | 2026-04-24 15:20:00 | 2398.90 | 2426.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2026-04-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:55:00 | 2480.00 | 2465.51 | 0.00 | ORB-long ORB[2447.60,2459.60] vol=1.6x ATR=5.40 |
| Stop hit — per-position SL triggered | 2026-04-29 11:15:00 | 2474.60 | 2468.35 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-05-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:55:00 | 2432.00 | 2448.80 | 0.00 | ORB-short ORB[2439.10,2469.00] vol=1.6x ATR=5.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 11:30:00 | 2423.74 | 2444.79 | 0.00 | T1 1.5R @ 2423.74 |
| Stop hit — per-position SL triggered | 2026-05-06 14:25:00 | 2432.00 | 2433.31 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-05-08 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:25:00 | 2384.80 | 2391.32 | 0.00 | ORB-short ORB[2388.70,2407.00] vol=1.5x ATR=5.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 11:15:00 | 2376.62 | 2389.10 | 0.00 | T1 1.5R @ 2376.62 |
| Stop hit — per-position SL triggered | 2026-05-08 13:05:00 | 2384.80 | 2386.85 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:30:00 | 2969.70 | 2026-02-10 09:40:00 | 2978.15 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2026-02-10 09:30:00 | 2969.70 | 2026-02-10 09:55:00 | 2969.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-18 10:35:00 | 2682.50 | 2026-02-18 12:35:00 | 2692.15 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-02-19 10:05:00 | 2745.40 | 2026-02-19 10:15:00 | 2737.06 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-11 10:35:00 | 2480.90 | 2026-03-11 11:00:00 | 2486.47 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-03-13 10:40:00 | 2441.00 | 2026-03-13 10:45:00 | 2434.66 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-03-16 10:20:00 | 2373.90 | 2026-03-16 11:15:00 | 2381.07 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-03-25 10:25:00 | 2415.00 | 2026-03-25 12:55:00 | 2408.55 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-03-30 09:30:00 | 2392.30 | 2026-03-30 09:50:00 | 2384.86 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-01 11:00:00 | 2445.50 | 2026-04-01 12:25:00 | 2437.21 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-02 09:40:00 | 2412.60 | 2026-04-02 09:45:00 | 2403.59 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-04-10 09:35:00 | 2516.80 | 2026-04-10 13:40:00 | 2529.01 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2026-04-21 10:55:00 | 2587.30 | 2026-04-21 11:35:00 | 2594.96 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-04-21 10:55:00 | 2587.30 | 2026-04-21 15:20:00 | 2611.60 | TARGET_HIT | 0.50 | 0.94% |
| SELL | retest1 | 2026-04-22 11:10:00 | 2554.60 | 2026-04-22 11:40:00 | 2546.98 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2026-04-22 11:10:00 | 2554.60 | 2026-04-22 13:55:00 | 2550.20 | TARGET_HIT | 0.50 | 0.17% |
| SELL | retest1 | 2026-04-24 09:45:00 | 2466.90 | 2026-04-24 10:10:00 | 2455.51 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-04-24 09:45:00 | 2466.90 | 2026-04-24 15:20:00 | 2398.90 | TARGET_HIT | 0.50 | 2.76% |
| BUY | retest1 | 2026-04-29 10:55:00 | 2480.00 | 2026-04-29 11:15:00 | 2474.60 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-05-06 10:55:00 | 2432.00 | 2026-05-06 11:30:00 | 2423.74 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-05-06 10:55:00 | 2432.00 | 2026-05-06 14:25:00 | 2432.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-08 10:25:00 | 2384.80 | 2026-05-08 11:15:00 | 2376.62 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-05-08 10:25:00 | 2384.80 | 2026-05-08 13:05:00 | 2384.80 | STOP_HIT | 0.50 | 0.00% |
