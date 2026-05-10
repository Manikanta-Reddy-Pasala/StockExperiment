# Siemens Energy India Ltd. (ENRIN)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 3186.00
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
| ENTRY1 | 10 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 5 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 5
- **Target hits / Stop hits / Partials:** 5 / 5 / 6
- **Avg / median % per leg:** 0.53% / 0.49%
- **Sum % (uncompounded):** 8.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 8 | 80.0% | 4 | 2 | 4 | 0.77% | 7.7% |
| BUY @ 2nd Alert (retest1) | 10 | 8 | 80.0% | 4 | 2 | 4 | 0.77% | 7.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.13% | 0.8% |
| SELL @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.13% | 0.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 16 | 11 | 68.8% | 5 | 5 | 6 | 0.53% | 8.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:50:00 | 2647.50 | 2660.28 | 0.00 | ORB-short ORB[2651.00,2689.30] vol=2.1x ATR=8.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 12:45:00 | 2634.47 | 2650.64 | 0.00 | T1 1.5R @ 2634.47 |
| Stop hit — per-position SL triggered | 2026-02-11 12:50:00 | 2647.50 | 2650.60 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 10:50:00 | 2735.70 | 2709.02 | 0.00 | ORB-long ORB[2686.00,2725.10] vol=2.0x ATR=10.40 |
| Stop hit — per-position SL triggered | 2026-02-13 11:00:00 | 2725.30 | 2710.34 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 10:05:00 | 2843.00 | 2827.26 | 0.00 | ORB-long ORB[2810.10,2838.70] vol=2.0x ATR=11.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 10:25:00 | 2860.39 | 2835.78 | 0.00 | T1 1.5R @ 2860.39 |
| Target hit | 2026-02-24 15:20:00 | 2908.90 | 2871.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 11:15:00 | 2855.00 | 2876.33 | 0.00 | ORB-short ORB[2863.00,2889.00] vol=2.4x ATR=9.79 |
| Stop hit — per-position SL triggered | 2026-03-06 12:30:00 | 2864.79 | 2872.59 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:55:00 | 2950.00 | 2965.06 | 0.00 | ORB-short ORB[2957.60,2990.00] vol=1.9x ATR=8.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:35:00 | 2937.48 | 2961.31 | 0.00 | T1 1.5R @ 2937.48 |
| Target hit | 2026-03-11 15:20:00 | 2934.00 | 2952.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 11:15:00 | 2739.00 | 2759.91 | 0.00 | ORB-short ORB[2759.50,2798.00] vol=1.9x ATR=8.45 |
| Stop hit — per-position SL triggered | 2026-03-17 12:45:00 | 2747.45 | 2753.83 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 10:15:00 | 2619.70 | 2598.94 | 0.00 | ORB-long ORB[2583.00,2614.90] vol=1.9x ATR=11.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-07 10:30:00 | 2637.16 | 2604.48 | 0.00 | T1 1.5R @ 2637.16 |
| Target hit | 2026-04-07 15:20:00 | 2628.90 | 2621.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2026-04-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:30:00 | 2926.20 | 2904.31 | 0.00 | ORB-long ORB[2888.00,2916.90] vol=1.5x ATR=11.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 09:55:00 | 2943.27 | 2915.78 | 0.00 | T1 1.5R @ 2943.27 |
| Target hit | 2026-04-17 15:20:00 | 3019.00 | 3007.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2026-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:30:00 | 3240.90 | 3213.59 | 0.00 | ORB-long ORB[3175.40,3222.10] vol=3.2x ATR=11.95 |
| Stop hit — per-position SL triggered | 2026-04-23 09:35:00 | 3228.95 | 3217.84 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-24 09:35:00 | 3246.50 | 3235.77 | 0.00 | ORB-long ORB[3205.50,3244.70] vol=2.6x ATR=13.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:00:00 | 3267.22 | 3245.95 | 0.00 | T1 1.5R @ 3267.22 |
| Target hit | 2026-04-24 10:25:00 | 3250.00 | 3253.37 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 09:50:00 | 2647.50 | 2026-02-11 12:45:00 | 2634.47 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-02-11 09:50:00 | 2647.50 | 2026-02-11 12:50:00 | 2647.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-13 10:50:00 | 2735.70 | 2026-02-13 11:00:00 | 2725.30 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-02-24 10:05:00 | 2843.00 | 2026-02-24 10:25:00 | 2860.39 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-02-24 10:05:00 | 2843.00 | 2026-02-24 15:20:00 | 2908.90 | TARGET_HIT | 0.50 | 2.32% |
| SELL | retest1 | 2026-03-06 11:15:00 | 2855.00 | 2026-03-06 12:30:00 | 2864.79 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-11 10:55:00 | 2950.00 | 2026-03-11 11:35:00 | 2937.48 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-03-11 10:55:00 | 2950.00 | 2026-03-11 15:20:00 | 2934.00 | TARGET_HIT | 0.50 | 0.54% |
| SELL | retest1 | 2026-03-17 11:15:00 | 2739.00 | 2026-03-17 12:45:00 | 2747.45 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-07 10:15:00 | 2619.70 | 2026-04-07 10:30:00 | 2637.16 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2026-04-07 10:15:00 | 2619.70 | 2026-04-07 15:20:00 | 2628.90 | TARGET_HIT | 0.50 | 0.35% |
| BUY | retest1 | 2026-04-17 09:30:00 | 2926.20 | 2026-04-17 09:55:00 | 2943.27 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-04-17 09:30:00 | 2926.20 | 2026-04-17 15:20:00 | 3019.00 | TARGET_HIT | 0.50 | 3.17% |
| BUY | retest1 | 2026-04-23 09:30:00 | 3240.90 | 2026-04-23 09:35:00 | 3228.95 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-24 09:35:00 | 3246.50 | 2026-04-24 10:00:00 | 3267.22 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2026-04-24 09:35:00 | 3246.50 | 2026-04-24 10:25:00 | 3250.00 | TARGET_HIT | 0.50 | 0.11% |
