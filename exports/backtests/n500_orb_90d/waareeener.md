# Waaree Energies Ltd. (WAAREEENER)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 3229.00
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
| ENTRY1 | 18 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 2 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 16
- **Target hits / Stop hits / Partials:** 2 / 16 / 7
- **Avg / median % per leg:** 0.06% / 0.00%
- **Sum % (uncompounded):** 1.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 7 | 46.7% | 2 | 8 | 5 | 0.15% | 2.3% |
| BUY @ 2nd Alert (retest1) | 15 | 7 | 46.7% | 2 | 8 | 5 | 0.15% | 2.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 2 | 20.0% | 0 | 8 | 2 | -0.08% | -0.8% |
| SELL @ 2nd Alert (retest1) | 10 | 2 | 20.0% | 0 | 8 | 2 | -0.08% | -0.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 25 | 9 | 36.0% | 2 | 16 | 7 | 0.06% | 1.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:55:00 | 3159.50 | 3142.85 | 0.00 | ORB-long ORB[3116.80,3155.00] vol=1.6x ATR=9.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:20:00 | 3173.80 | 3151.00 | 0.00 | T1 1.5R @ 3173.80 |
| Stop hit — per-position SL triggered | 2026-02-10 11:10:00 | 3159.50 | 3157.97 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:15:00 | 3136.60 | 3149.27 | 0.00 | ORB-short ORB[3143.20,3180.90] vol=2.5x ATR=7.29 |
| Stop hit — per-position SL triggered | 2026-02-12 13:15:00 | 3143.89 | 3145.58 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 10:40:00 | 3094.10 | 3114.29 | 0.00 | ORB-short ORB[3102.00,3127.00] vol=1.9x ATR=8.13 |
| Stop hit — per-position SL triggered | 2026-02-17 10:45:00 | 3102.23 | 3113.65 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:20:00 | 2911.00 | 2942.45 | 0.00 | ORB-short ORB[2945.80,2988.40] vol=1.6x ATR=14.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:00:00 | 2889.85 | 2929.12 | 0.00 | T1 1.5R @ 2889.85 |
| Stop hit — per-position SL triggered | 2026-02-19 11:30:00 | 2911.00 | 2927.11 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 11:15:00 | 2888.70 | 2903.77 | 0.00 | ORB-short ORB[2908.50,2934.60] vol=1.5x ATR=9.50 |
| Stop hit — per-position SL triggered | 2026-02-23 12:05:00 | 2898.20 | 2901.54 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 09:50:00 | 2732.60 | 2706.26 | 0.00 | ORB-long ORB[2687.00,2724.50] vol=1.8x ATR=11.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:20:00 | 2749.64 | 2719.87 | 0.00 | T1 1.5R @ 2749.64 |
| Target hit | 2026-02-27 12:35:00 | 2734.90 | 2735.21 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — SELL (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:15:00 | 2608.20 | 2636.24 | 0.00 | ORB-short ORB[2650.00,2671.60] vol=2.1x ATR=8.92 |
| Stop hit — per-position SL triggered | 2026-03-05 12:05:00 | 2617.12 | 2630.35 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:55:00 | 2712.60 | 2715.21 | 0.00 | ORB-short ORB[2717.00,2752.90] vol=1.6x ATR=11.18 |
| Stop hit — per-position SL triggered | 2026-03-13 11:10:00 | 2723.78 | 2719.02 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:40:00 | 2864.50 | 2822.54 | 0.00 | ORB-long ORB[2786.20,2823.70] vol=2.5x ATR=18.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 10:05:00 | 2891.99 | 2847.97 | 0.00 | T1 1.5R @ 2891.99 |
| Target hit | 2026-03-17 11:55:00 | 2896.10 | 2901.90 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 2954.20 | 2920.03 | 0.00 | ORB-long ORB[2890.00,2925.00] vol=2.6x ATR=13.70 |
| Stop hit — per-position SL triggered | 2026-03-18 09:35:00 | 2940.50 | 2927.56 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 11:00:00 | 3173.50 | 3149.82 | 0.00 | ORB-long ORB[3128.10,3163.00] vol=1.5x ATR=10.61 |
| Stop hit — per-position SL triggered | 2026-03-25 11:15:00 | 3162.89 | 3152.25 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-02 10:45:00 | 2989.50 | 3036.37 | 0.00 | ORB-short ORB[3022.70,3066.00] vol=1.9x ATR=14.21 |
| Stop hit — per-position SL triggered | 2026-04-02 11:25:00 | 3003.71 | 3029.75 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:35:00 | 3432.00 | 3453.29 | 0.00 | ORB-short ORB[3435.00,3468.00] vol=1.7x ATR=11.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 09:45:00 | 3414.41 | 3444.40 | 0.00 | T1 1.5R @ 3414.41 |
| Stop hit — per-position SL triggered | 2026-04-16 10:10:00 | 3432.00 | 3435.89 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:45:00 | 3503.00 | 3491.40 | 0.00 | ORB-long ORB[3456.60,3499.00] vol=2.1x ATR=13.49 |
| Stop hit — per-position SL triggered | 2026-04-17 10:25:00 | 3489.51 | 3493.25 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:30:00 | 3538.60 | 3525.25 | 0.00 | ORB-long ORB[3492.70,3533.50] vol=4.1x ATR=11.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:15:00 | 3555.57 | 3535.20 | 0.00 | T1 1.5R @ 3555.57 |
| Stop hit — per-position SL triggered | 2026-04-21 10:20:00 | 3538.60 | 3535.24 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-05-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:45:00 | 3158.90 | 3143.93 | 0.00 | ORB-long ORB[3116.20,3152.80] vol=2.2x ATR=10.90 |
| Stop hit — per-position SL triggered | 2026-05-05 10:00:00 | 3148.00 | 3145.74 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-05-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:00:00 | 3213.00 | 3197.63 | 0.00 | ORB-long ORB[3186.10,3210.00] vol=2.1x ATR=8.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:10:00 | 3226.16 | 3205.88 | 0.00 | T1 1.5R @ 3226.16 |
| Stop hit — per-position SL triggered | 2026-05-06 10:30:00 | 3213.00 | 3214.35 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-05-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:30:00 | 3284.10 | 3264.70 | 0.00 | ORB-long ORB[3231.30,3280.00] vol=2.5x ATR=10.86 |
| Stop hit — per-position SL triggered | 2026-05-08 09:35:00 | 3273.24 | 3265.84 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:55:00 | 3159.50 | 2026-02-10 10:20:00 | 3173.80 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-10 09:55:00 | 3159.50 | 2026-02-10 11:10:00 | 3159.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-12 11:15:00 | 3136.60 | 2026-02-12 13:15:00 | 3143.89 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-02-17 10:40:00 | 3094.10 | 2026-02-17 10:45:00 | 3102.23 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-19 10:20:00 | 2911.00 | 2026-02-19 11:00:00 | 2889.85 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2026-02-19 10:20:00 | 2911.00 | 2026-02-19 11:30:00 | 2911.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-23 11:15:00 | 2888.70 | 2026-02-23 12:05:00 | 2898.20 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-27 09:50:00 | 2732.60 | 2026-02-27 10:20:00 | 2749.64 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-02-27 09:50:00 | 2732.60 | 2026-02-27 12:35:00 | 2734.90 | TARGET_HIT | 0.50 | 0.08% |
| SELL | retest1 | 2026-03-05 11:15:00 | 2608.20 | 2026-03-05 12:05:00 | 2617.12 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-13 10:55:00 | 2712.60 | 2026-03-13 11:10:00 | 2723.78 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-03-17 09:40:00 | 2864.50 | 2026-03-17 10:05:00 | 2891.99 | PARTIAL | 0.50 | 0.96% |
| BUY | retest1 | 2026-03-17 09:40:00 | 2864.50 | 2026-03-17 11:55:00 | 2896.10 | TARGET_HIT | 0.50 | 1.10% |
| BUY | retest1 | 2026-03-18 09:30:00 | 2954.20 | 2026-03-18 09:35:00 | 2940.50 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2026-03-25 11:00:00 | 3173.50 | 2026-03-25 11:15:00 | 3162.89 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-04-02 10:45:00 | 2989.50 | 2026-04-02 11:25:00 | 3003.71 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2026-04-16 09:35:00 | 3432.00 | 2026-04-16 09:45:00 | 3414.41 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-04-16 09:35:00 | 3432.00 | 2026-04-16 10:10:00 | 3432.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 09:45:00 | 3503.00 | 2026-04-17 10:25:00 | 3489.51 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-04-21 09:30:00 | 3538.60 | 2026-04-21 10:15:00 | 3555.57 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-04-21 09:30:00 | 3538.60 | 2026-04-21 10:20:00 | 3538.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-05 09:45:00 | 3158.90 | 2026-05-05 10:00:00 | 3148.00 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-05-06 10:00:00 | 3213.00 | 2026-05-06 10:10:00 | 3226.16 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-05-06 10:00:00 | 3213.00 | 2026-05-06 10:30:00 | 3213.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-08 09:30:00 | 3284.10 | 2026-05-08 09:35:00 | 3273.24 | STOP_HIT | 1.00 | -0.33% |
