# Godrej Properties Ltd. (GODREJPROP)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2024-08-08 15:25:00 (4596 bars)
- **Last close:** 2878.00
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
| ENTRY1 | 22 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 5 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 17
- **Target hits / Stop hits / Partials:** 5 / 17 / 6
- **Avg / median % per leg:** 0.24% / -0.28%
- **Sum % (uncompounded):** 6.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 11 | 52.4% | 5 | 10 | 6 | 0.46% | 9.6% |
| BUY @ 2nd Alert (retest1) | 21 | 11 | 52.4% | 5 | 10 | 6 | 0.46% | 9.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -0.40% | -2.8% |
| SELL @ 2nd Alert (retest1) | 7 | 0 | 0.0% | 0 | 7 | 0 | -0.40% | -2.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 28 | 11 | 39.3% | 5 | 17 | 6 | 0.24% | 6.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 09:50:00 | 2858.45 | 2829.75 | 0.00 | ORB-long ORB[2790.00,2827.00] vol=3.2x ATR=12.11 |
| Stop hit — per-position SL triggered | 2024-05-15 10:05:00 | 2846.34 | 2839.56 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-06-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 11:10:00 | 2841.00 | 2809.92 | 0.00 | ORB-long ORB[2782.05,2822.00] vol=4.7x ATR=9.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-07 11:55:00 | 2854.97 | 2820.94 | 0.00 | T1 1.5R @ 2854.97 |
| Target hit | 2024-06-07 15:20:00 | 2878.80 | 2847.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2024-06-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 11:10:00 | 2878.90 | 2865.84 | 0.00 | ORB-long ORB[2830.00,2860.35] vol=1.6x ATR=9.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 11:15:00 | 2892.70 | 2867.68 | 0.00 | T1 1.5R @ 2892.70 |
| Target hit | 2024-06-12 15:20:00 | 2924.90 | 2906.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2024-06-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 09:35:00 | 2984.90 | 2966.31 | 0.00 | ORB-long ORB[2935.00,2978.00] vol=1.9x ATR=12.36 |
| Stop hit — per-position SL triggered | 2024-06-13 09:40:00 | 2972.54 | 2967.40 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 11:05:00 | 3024.80 | 3008.33 | 0.00 | ORB-long ORB[2996.30,3023.35] vol=3.5x ATR=10.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 11:30:00 | 3040.10 | 3014.15 | 0.00 | T1 1.5R @ 3040.10 |
| Stop hit — per-position SL triggered | 2024-06-18 11:50:00 | 3024.80 | 3015.65 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 10:35:00 | 3041.25 | 3017.55 | 0.00 | ORB-long ORB[2982.05,3018.55] vol=1.7x ATR=9.94 |
| Stop hit — per-position SL triggered | 2024-06-20 11:05:00 | 3031.31 | 3023.40 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 10:50:00 | 3023.45 | 3015.97 | 0.00 | ORB-long ORB[2995.00,3019.95] vol=2.9x ATR=9.12 |
| Stop hit — per-position SL triggered | 2024-06-21 12:05:00 | 3014.33 | 3017.54 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 09:35:00 | 3000.00 | 2985.03 | 0.00 | ORB-long ORB[2956.50,2999.25] vol=1.6x ATR=12.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-24 10:20:00 | 3018.77 | 2997.49 | 0.00 | T1 1.5R @ 3018.77 |
| Target hit | 2024-06-24 15:20:00 | 3106.00 | 3066.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2024-06-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 10:35:00 | 3063.10 | 3057.54 | 0.00 | ORB-long ORB[3026.05,3055.95] vol=1.5x ATR=8.58 |
| Stop hit — per-position SL triggered | 2024-06-27 10:45:00 | 3054.52 | 3057.51 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 09:55:00 | 3239.45 | 3210.28 | 0.00 | ORB-long ORB[3171.10,3214.80] vol=1.7x ATR=13.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 10:00:00 | 3259.44 | 3218.01 | 0.00 | T1 1.5R @ 3259.44 |
| Target hit | 2024-07-02 15:20:00 | 3296.35 | 3293.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2024-07-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-03 09:35:00 | 3277.75 | 3289.32 | 0.00 | ORB-short ORB[3282.30,3319.80] vol=4.8x ATR=12.23 |
| Stop hit — per-position SL triggered | 2024-07-03 09:40:00 | 3289.98 | 3289.09 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 10:30:00 | 3304.00 | 3319.98 | 0.00 | ORB-short ORB[3305.70,3335.80] vol=1.7x ATR=10.51 |
| Stop hit — per-position SL triggered | 2024-07-04 11:05:00 | 3314.51 | 3317.28 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 09:30:00 | 3255.75 | 3272.94 | 0.00 | ORB-short ORB[3270.00,3308.00] vol=5.7x ATR=11.81 |
| Stop hit — per-position SL triggered | 2024-07-05 09:40:00 | 3267.56 | 3270.82 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 10:30:00 | 3275.00 | 3282.31 | 0.00 | ORB-short ORB[3276.75,3325.00] vol=4.9x ATR=11.16 |
| Stop hit — per-position SL triggered | 2024-07-08 10:35:00 | 3286.16 | 3282.32 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-10 09:30:00 | 3315.20 | 3301.03 | 0.00 | ORB-long ORB[3269.20,3306.75] vol=3.3x ATR=9.33 |
| Stop hit — per-position SL triggered | 2024-07-10 09:45:00 | 3305.87 | 3307.25 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 10:50:00 | 3295.15 | 3305.11 | 0.00 | ORB-short ORB[3300.85,3327.00] vol=7.5x ATR=8.93 |
| Stop hit — per-position SL triggered | 2024-07-11 11:00:00 | 3304.08 | 3304.96 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 11:00:00 | 3312.05 | 3292.27 | 0.00 | ORB-long ORB[3275.40,3305.00] vol=2.9x ATR=9.20 |
| Stop hit — per-position SL triggered | 2024-07-12 11:30:00 | 3302.85 | 3301.14 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 11:00:00 | 3269.85 | 3319.89 | 0.00 | ORB-short ORB[3325.10,3372.65] vol=1.8x ATR=12.00 |
| Stop hit — per-position SL triggered | 2024-07-19 11:25:00 | 3281.85 | 3314.05 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-24 09:45:00 | 3092.55 | 3104.86 | 0.00 | ORB-short ORB[3107.50,3153.45] vol=1.5x ATR=23.99 |
| Stop hit — per-position SL triggered | 2024-07-24 10:10:00 | 3116.54 | 3102.73 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:30:00 | 3124.55 | 3101.96 | 0.00 | ORB-long ORB[3080.80,3124.00] vol=2.7x ATR=11.62 |
| Stop hit — per-position SL triggered | 2024-07-26 09:40:00 | 3112.93 | 3108.55 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-07-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 09:40:00 | 3174.00 | 3153.47 | 0.00 | ORB-long ORB[3127.00,3158.85] vol=2.6x ATR=11.72 |
| Stop hit — per-position SL triggered | 2024-07-29 09:50:00 | 3162.28 | 3154.30 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-07-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 10:55:00 | 3147.95 | 3128.85 | 0.00 | ORB-long ORB[3100.40,3136.15] vol=1.9x ATR=9.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 11:05:00 | 3162.30 | 3144.71 | 0.00 | T1 1.5R @ 3162.30 |
| Target hit | 2024-07-30 15:20:00 | 3186.10 | 3166.39 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-15 09:50:00 | 2858.45 | 2024-05-15 10:05:00 | 2846.34 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-06-07 11:10:00 | 2841.00 | 2024-06-07 11:55:00 | 2854.97 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-06-07 11:10:00 | 2841.00 | 2024-06-07 15:20:00 | 2878.80 | TARGET_HIT | 0.50 | 1.33% |
| BUY | retest1 | 2024-06-12 11:10:00 | 2878.90 | 2024-06-12 11:15:00 | 2892.70 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-06-12 11:10:00 | 2878.90 | 2024-06-12 15:20:00 | 2924.90 | TARGET_HIT | 0.50 | 1.60% |
| BUY | retest1 | 2024-06-13 09:35:00 | 2984.90 | 2024-06-13 09:40:00 | 2972.54 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-06-18 11:05:00 | 3024.80 | 2024-06-18 11:30:00 | 3040.10 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-06-18 11:05:00 | 3024.80 | 2024-06-18 11:50:00 | 3024.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-20 10:35:00 | 3041.25 | 2024-06-20 11:05:00 | 3031.31 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-06-21 10:50:00 | 3023.45 | 2024-06-21 12:05:00 | 3014.33 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-06-24 09:35:00 | 3000.00 | 2024-06-24 10:20:00 | 3018.77 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-06-24 09:35:00 | 3000.00 | 2024-06-24 15:20:00 | 3106.00 | TARGET_HIT | 0.50 | 3.53% |
| BUY | retest1 | 2024-06-27 10:35:00 | 3063.10 | 2024-06-27 10:45:00 | 3054.52 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-02 09:55:00 | 3239.45 | 2024-07-02 10:00:00 | 3259.44 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-07-02 09:55:00 | 3239.45 | 2024-07-02 15:20:00 | 3296.35 | TARGET_HIT | 0.50 | 1.76% |
| SELL | retest1 | 2024-07-03 09:35:00 | 3277.75 | 2024-07-03 09:40:00 | 3289.98 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-07-04 10:30:00 | 3304.00 | 2024-07-04 11:05:00 | 3314.51 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-07-05 09:30:00 | 3255.75 | 2024-07-05 09:40:00 | 3267.56 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-07-08 10:30:00 | 3275.00 | 2024-07-08 10:35:00 | 3286.16 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-07-10 09:30:00 | 3315.20 | 2024-07-10 09:45:00 | 3305.87 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-07-11 10:50:00 | 3295.15 | 2024-07-11 11:00:00 | 3304.08 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-07-12 11:00:00 | 3312.05 | 2024-07-12 11:30:00 | 3302.85 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-07-19 11:00:00 | 3269.85 | 2024-07-19 11:25:00 | 3281.85 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-07-24 09:45:00 | 3092.55 | 2024-07-24 10:10:00 | 3116.54 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest1 | 2024-07-26 09:30:00 | 3124.55 | 2024-07-26 09:40:00 | 3112.93 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-07-29 09:40:00 | 3174.00 | 2024-07-29 09:50:00 | 3162.28 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-07-30 10:55:00 | 3147.95 | 2024-07-30 11:05:00 | 3162.30 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-07-30 10:55:00 | 3147.95 | 2024-07-30 15:20:00 | 3186.10 | TARGET_HIT | 0.50 | 1.21% |
