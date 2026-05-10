# Tube Investments of India Ltd. (TIINDIA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 3032.70
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
| ENTRY1 | 15 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 5 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 10
- **Target hits / Stop hits / Partials:** 5 / 10 / 7
- **Avg / median % per leg:** 0.34% / 0.46%
- **Sum % (uncompounded):** 7.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 7 | 63.6% | 3 | 4 | 4 | 0.56% | 6.1% |
| BUY @ 2nd Alert (retest1) | 11 | 7 | 63.6% | 3 | 4 | 4 | 0.56% | 6.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 5 | 45.5% | 2 | 6 | 3 | 0.13% | 1.4% |
| SELL @ 2nd Alert (retest1) | 11 | 5 | 45.5% | 2 | 6 | 3 | 0.13% | 1.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 12 | 54.5% | 5 | 10 | 7 | 0.34% | 7.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:30:00 | 2394.40 | 2378.12 | 0.00 | ORB-long ORB[2356.50,2391.10] vol=1.8x ATR=10.28 |
| Stop hit — per-position SL triggered | 2026-02-10 10:35:00 | 2384.12 | 2378.38 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:50:00 | 2460.00 | 2437.08 | 0.00 | ORB-long ORB[2414.20,2446.00] vol=3.3x ATR=7.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:00:00 | 2471.24 | 2443.82 | 0.00 | T1 1.5R @ 2471.24 |
| Target hit | 2026-02-12 15:20:00 | 2506.00 | 2485.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-02-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:35:00 | 2512.30 | 2488.53 | 0.00 | ORB-long ORB[2477.20,2496.00] vol=1.8x ATR=9.36 |
| Stop hit — per-position SL triggered | 2026-02-17 09:40:00 | 2502.94 | 2489.27 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:55:00 | 2434.00 | 2443.53 | 0.00 | ORB-short ORB[2445.00,2477.60] vol=4.5x ATR=7.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:10:00 | 2423.21 | 2439.19 | 0.00 | T1 1.5R @ 2423.21 |
| Stop hit — per-position SL triggered | 2026-02-18 11:25:00 | 2434.00 | 2437.47 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 11:00:00 | 2556.30 | 2579.42 | 0.00 | ORB-short ORB[2570.00,2599.10] vol=2.0x ATR=11.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:45:00 | 2539.42 | 2575.96 | 0.00 | T1 1.5R @ 2539.42 |
| Target hit | 2026-02-23 15:20:00 | 2540.10 | 2559.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2026-02-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 11:05:00 | 2612.60 | 2595.67 | 0.00 | ORB-long ORB[2575.00,2610.00] vol=4.4x ATR=7.79 |
| Stop hit — per-position SL triggered | 2026-02-25 12:35:00 | 2604.81 | 2602.52 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:20:00 | 2661.30 | 2635.03 | 0.00 | ORB-long ORB[2606.50,2617.90] vol=4.2x ATR=10.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:25:00 | 2676.97 | 2649.88 | 0.00 | T1 1.5R @ 2676.97 |
| Stop hit — per-position SL triggered | 2026-02-26 10:30:00 | 2661.30 | 2652.21 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 09:55:00 | 2414.00 | 2430.48 | 0.00 | ORB-short ORB[2433.30,2459.10] vol=3.1x ATR=10.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:15:00 | 2397.98 | 2424.89 | 0.00 | T1 1.5R @ 2397.98 |
| Target hit | 2026-03-16 11:15:00 | 2396.10 | 2391.87 | 0.00 | Trail-exit close>VWAP |

### Cycle 9 — BUY (started 2026-03-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 11:00:00 | 2515.30 | 2479.17 | 0.00 | ORB-long ORB[2447.60,2478.60] vol=4.0x ATR=10.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 11:20:00 | 2530.37 | 2495.59 | 0.00 | T1 1.5R @ 2530.37 |
| Target hit | 2026-03-18 15:20:00 | 2556.00 | 2527.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2026-04-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 10:50:00 | 2737.80 | 2750.38 | 0.00 | ORB-short ORB[2742.60,2768.50] vol=7.0x ATR=8.80 |
| Stop hit — per-position SL triggered | 2026-04-15 11:20:00 | 2746.60 | 2749.96 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 09:50:00 | 2739.10 | 2750.85 | 0.00 | ORB-short ORB[2744.10,2782.40] vol=1.6x ATR=11.41 |
| Stop hit — per-position SL triggered | 2026-04-17 09:55:00 | 2750.51 | 2749.63 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-28 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 10:10:00 | 2961.80 | 2966.43 | 0.00 | ORB-short ORB[2970.00,3004.30] vol=5.1x ATR=8.96 |
| Stop hit — per-position SL triggered | 2026-04-28 10:15:00 | 2970.76 | 2972.31 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-05-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 10:25:00 | 2928.80 | 2939.81 | 0.00 | ORB-short ORB[2956.00,2978.60] vol=1.6x ATR=11.23 |
| Stop hit — per-position SL triggered | 2026-05-04 10:50:00 | 2940.03 | 2939.37 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-05-05 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:30:00 | 2898.50 | 2910.71 | 0.00 | ORB-short ORB[2905.10,2920.00] vol=1.9x ATR=8.28 |
| Stop hit — per-position SL triggered | 2026-05-05 11:35:00 | 2906.78 | 2907.04 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-05-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:10:00 | 2980.30 | 2965.94 | 0.00 | ORB-long ORB[2928.50,2965.00] vol=2.0x ATR=14.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 13:35:00 | 3001.43 | 2978.40 | 0.00 | T1 1.5R @ 3001.43 |
| Target hit | 2026-05-07 15:20:00 | 3021.90 | 2992.04 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 10:30:00 | 2394.40 | 2026-02-10 10:35:00 | 2384.12 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-02-12 10:50:00 | 2460.00 | 2026-02-12 11:00:00 | 2471.24 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-02-12 10:50:00 | 2460.00 | 2026-02-12 15:20:00 | 2506.00 | TARGET_HIT | 0.50 | 1.87% |
| BUY | retest1 | 2026-02-17 09:35:00 | 2512.30 | 2026-02-17 09:40:00 | 2502.94 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-18 10:55:00 | 2434.00 | 2026-02-18 11:10:00 | 2423.21 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-02-18 10:55:00 | 2434.00 | 2026-02-18 11:25:00 | 2434.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-23 11:00:00 | 2556.30 | 2026-02-23 11:45:00 | 2539.42 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2026-02-23 11:00:00 | 2556.30 | 2026-02-23 15:20:00 | 2540.10 | TARGET_HIT | 0.50 | 0.63% |
| BUY | retest1 | 2026-02-25 11:05:00 | 2612.60 | 2026-02-25 12:35:00 | 2604.81 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-26 10:20:00 | 2661.30 | 2026-02-26 10:25:00 | 2676.97 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2026-02-26 10:20:00 | 2661.30 | 2026-02-26 10:30:00 | 2661.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-16 09:55:00 | 2414.00 | 2026-03-16 10:15:00 | 2397.98 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2026-03-16 09:55:00 | 2414.00 | 2026-03-16 11:15:00 | 2396.10 | TARGET_HIT | 0.50 | 0.74% |
| BUY | retest1 | 2026-03-18 11:00:00 | 2515.30 | 2026-03-18 11:20:00 | 2530.37 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-03-18 11:00:00 | 2515.30 | 2026-03-18 15:20:00 | 2556.00 | TARGET_HIT | 0.50 | 1.62% |
| SELL | retest1 | 2026-04-15 10:50:00 | 2737.80 | 2026-04-15 11:20:00 | 2746.60 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-04-17 09:50:00 | 2739.10 | 2026-04-17 09:55:00 | 2750.51 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-04-28 10:10:00 | 2961.80 | 2026-04-28 10:15:00 | 2970.76 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-05-04 10:25:00 | 2928.80 | 2026-05-04 10:50:00 | 2940.03 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-05-05 10:30:00 | 2898.50 | 2026-05-05 11:35:00 | 2906.78 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-05-07 10:10:00 | 2980.30 | 2026-05-07 13:35:00 | 3001.43 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2026-05-07 10:10:00 | 2980.30 | 2026-05-07 15:20:00 | 3021.90 | TARGET_HIT | 0.50 | 1.40% |
