# PI Industries Ltd. (PIIND)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 3103.60
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
| ENTRY1 | 16 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 15
- **Target hits / Stop hits / Partials:** 1 / 15 / 4
- **Avg / median % per leg:** -0.08% / -0.21%
- **Sum % (uncompounded):** -1.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 3 | 27.3% | 1 | 8 | 2 | -0.10% | -1.1% |
| BUY @ 2nd Alert (retest1) | 11 | 3 | 27.3% | 1 | 8 | 2 | -0.10% | -1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 2 | 22.2% | 0 | 7 | 2 | -0.06% | -0.6% |
| SELL @ 2nd Alert (retest1) | 9 | 2 | 22.2% | 0 | 7 | 2 | -0.06% | -0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 20 | 5 | 25.0% | 1 | 15 | 4 | -0.08% | -1.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:30:00 | 3152.40 | 3143.59 | 0.00 | ORB-long ORB[3123.80,3140.70] vol=5.2x ATR=9.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 10:45:00 | 3166.23 | 3145.82 | 0.00 | T1 1.5R @ 3166.23 |
| Stop hit — per-position SL triggered | 2026-02-09 10:50:00 | 3152.40 | 3146.01 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 11:00:00 | 3182.20 | 3191.33 | 0.00 | ORB-short ORB[3188.00,3230.00] vol=6.8x ATR=9.76 |
| Stop hit — per-position SL triggered | 2026-02-10 11:20:00 | 3191.96 | 3189.73 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:50:00 | 3126.90 | 3119.40 | 0.00 | ORB-long ORB[3094.10,3114.90] vol=2.3x ATR=5.37 |
| Stop hit — per-position SL triggered | 2026-02-26 11:00:00 | 3121.53 | 3120.20 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:00:00 | 2909.00 | 2928.48 | 0.00 | ORB-short ORB[2931.10,2958.00] vol=1.8x ATR=8.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:10:00 | 2895.76 | 2925.13 | 0.00 | T1 1.5R @ 2895.76 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 2909.00 | 2913.42 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:45:00 | 2859.90 | 2868.97 | 0.00 | ORB-short ORB[2865.80,2900.00] vol=3.6x ATR=10.63 |
| Stop hit — per-position SL triggered | 2026-03-16 10:55:00 | 2870.53 | 2868.42 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:55:00 | 2907.40 | 2893.87 | 0.00 | ORB-long ORB[2882.00,2901.60] vol=1.6x ATR=10.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 10:10:00 | 2922.70 | 2898.04 | 0.00 | T1 1.5R @ 2922.70 |
| Target hit | 2026-03-17 13:20:00 | 2912.00 | 2914.43 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — BUY (started 2026-03-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 11:00:00 | 2918.40 | 2901.07 | 0.00 | ORB-long ORB[2866.30,2895.10] vol=9.8x ATR=8.28 |
| Stop hit — per-position SL triggered | 2026-03-18 11:10:00 | 2910.12 | 2908.44 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 11:05:00 | 2826.90 | 2854.17 | 0.00 | ORB-short ORB[2860.20,2897.00] vol=2.3x ATR=9.19 |
| Stop hit — per-position SL triggered | 2026-03-27 11:10:00 | 2836.09 | 2851.47 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:50:00 | 2828.30 | 2856.42 | 0.00 | ORB-short ORB[2844.30,2885.40] vol=1.6x ATR=8.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 10:10:00 | 2814.98 | 2832.59 | 0.00 | T1 1.5R @ 2814.98 |
| Stop hit — per-position SL triggered | 2026-04-09 10:30:00 | 2828.30 | 2829.39 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-24 09:50:00 | 3103.00 | 3088.60 | 0.00 | ORB-long ORB[3071.10,3098.90] vol=2.9x ATR=11.18 |
| Stop hit — per-position SL triggered | 2026-04-24 09:55:00 | 3091.82 | 3090.23 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:40:00 | 3116.40 | 3104.04 | 0.00 | ORB-long ORB[3080.00,3115.40] vol=1.6x ATR=10.71 |
| Stop hit — per-position SL triggered | 2026-04-27 10:50:00 | 3105.69 | 3104.31 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-28 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:10:00 | 3138.00 | 3100.14 | 0.00 | ORB-long ORB[3072.40,3100.00] vol=4.1x ATR=11.02 |
| Stop hit — per-position SL triggered | 2026-04-28 10:55:00 | 3126.98 | 3122.36 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:45:00 | 3027.70 | 3042.59 | 0.00 | ORB-short ORB[3051.60,3090.00] vol=4.4x ATR=8.83 |
| Stop hit — per-position SL triggered | 2026-04-30 13:05:00 | 3036.53 | 3035.92 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-05-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:50:00 | 3005.00 | 3011.74 | 0.00 | ORB-short ORB[3007.00,3035.00] vol=2.6x ATR=6.33 |
| Stop hit — per-position SL triggered | 2026-05-05 11:10:00 | 3011.33 | 3011.11 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-05-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:30:00 | 3080.30 | 3061.93 | 0.00 | ORB-long ORB[3021.10,3058.40] vol=2.1x ATR=9.93 |
| Stop hit — per-position SL triggered | 2026-05-06 11:10:00 | 3070.37 | 3073.47 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-05-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:00:00 | 3115.10 | 3095.64 | 0.00 | ORB-long ORB[3072.50,3107.50] vol=2.1x ATR=11.11 |
| Stop hit — per-position SL triggered | 2026-05-07 10:35:00 | 3103.99 | 3099.70 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:30:00 | 3152.40 | 2026-02-09 10:45:00 | 3166.23 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-02-09 10:30:00 | 3152.40 | 2026-02-09 10:50:00 | 3152.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-10 11:00:00 | 3182.20 | 2026-02-10 11:20:00 | 3191.96 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-26 10:50:00 | 3126.90 | 2026-02-26 11:00:00 | 3121.53 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2026-03-13 10:00:00 | 2909.00 | 2026-03-13 10:10:00 | 2895.76 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-03-13 10:00:00 | 2909.00 | 2026-03-13 10:50:00 | 2909.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-16 10:45:00 | 2859.90 | 2026-03-16 10:55:00 | 2870.53 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-03-17 09:55:00 | 2907.40 | 2026-03-17 10:10:00 | 2922.70 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-03-17 09:55:00 | 2907.40 | 2026-03-17 13:20:00 | 2912.00 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2026-03-18 11:00:00 | 2918.40 | 2026-03-18 11:10:00 | 2910.12 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-03-27 11:05:00 | 2826.90 | 2026-03-27 11:10:00 | 2836.09 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-04-09 09:50:00 | 2828.30 | 2026-04-09 10:10:00 | 2814.98 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-04-09 09:50:00 | 2828.30 | 2026-04-09 10:30:00 | 2828.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-24 09:50:00 | 3103.00 | 2026-04-24 09:55:00 | 3091.82 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-27 10:40:00 | 3116.40 | 2026-04-27 10:50:00 | 3105.69 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-28 10:10:00 | 3138.00 | 2026-04-28 10:55:00 | 3126.98 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-30 10:45:00 | 3027.70 | 2026-04-30 13:05:00 | 3036.53 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-05-05 10:50:00 | 3005.00 | 2026-05-05 11:10:00 | 3011.33 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-05-06 10:30:00 | 3080.30 | 2026-05-06 11:10:00 | 3070.37 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-05-07 10:00:00 | 3115.10 | 2026-05-07 10:35:00 | 3103.99 | STOP_HIT | 1.00 | -0.36% |
