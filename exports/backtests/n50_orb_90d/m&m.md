# M&M (M&M)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 3331.50
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
| TARGET_HIT | 2 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 13
- **Target hits / Stop hits / Partials:** 2 / 13 / 7
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 3.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 5 | 41.7% | 1 | 7 | 4 | 0.05% | 0.6% |
| BUY @ 2nd Alert (retest1) | 12 | 5 | 41.7% | 1 | 7 | 4 | 0.05% | 0.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.25% | 2.5% |
| SELL @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.25% | 2.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 9 | 40.9% | 2 | 13 | 7 | 0.14% | 3.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 3630.80 | 3622.37 | 0.00 | ORB-long ORB[3601.00,3629.70] vol=1.6x ATR=6.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:40:00 | 3640.82 | 3626.56 | 0.00 | T1 1.5R @ 3640.82 |
| Stop hit — per-position SL triggered | 2026-02-10 09:50:00 | 3630.80 | 3627.79 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 09:45:00 | 3746.00 | 3724.19 | 0.00 | ORB-long ORB[3678.00,3729.70] vol=1.9x ATR=10.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 10:20:00 | 3761.21 | 3732.45 | 0.00 | T1 1.5R @ 3761.21 |
| Stop hit — per-position SL triggered | 2026-02-11 11:50:00 | 3746.00 | 3746.51 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 11:15:00 | 3475.70 | 3478.23 | 0.00 | ORB-short ORB[3476.90,3515.80] vol=2.8x ATR=7.41 |
| Stop hit — per-position SL triggered | 2026-02-17 11:55:00 | 3483.11 | 3478.09 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:45:00 | 3489.00 | 3513.87 | 0.00 | ORB-short ORB[3523.30,3546.50] vol=4.4x ATR=7.08 |
| Stop hit — per-position SL triggered | 2026-02-19 10:50:00 | 3496.08 | 3513.19 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 11:00:00 | 3488.90 | 3461.05 | 0.00 | ORB-long ORB[3425.40,3455.00] vol=1.5x ATR=7.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:50:00 | 3500.14 | 3466.75 | 0.00 | T1 1.5R @ 3500.14 |
| Stop hit — per-position SL triggered | 2026-02-25 12:10:00 | 3488.90 | 3468.46 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:30:00 | 3242.50 | 3267.22 | 0.00 | ORB-short ORB[3260.50,3292.90] vol=2.5x ATR=10.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 10:35:00 | 3227.27 | 3247.89 | 0.00 | T1 1.5R @ 3227.27 |
| Target hit | 2026-03-11 15:20:00 | 3175.70 | 3207.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 11:15:00 | 3104.10 | 3072.72 | 0.00 | ORB-long ORB[3056.80,3083.00] vol=4.7x ATR=11.11 |
| Stop hit — per-position SL triggered | 2026-03-20 11:30:00 | 3092.99 | 3074.76 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 09:35:00 | 3189.70 | 3207.41 | 0.00 | ORB-short ORB[3190.00,3222.80] vol=1.8x ATR=13.17 |
| Stop hit — per-position SL triggered | 2026-04-13 09:40:00 | 3202.87 | 3206.66 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:50:00 | 3245.20 | 3263.31 | 0.00 | ORB-short ORB[3260.00,3288.00] vol=1.6x ATR=9.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 10:25:00 | 3230.86 | 3253.77 | 0.00 | T1 1.5R @ 3230.86 |
| Stop hit — per-position SL triggered | 2026-04-16 10:45:00 | 3245.20 | 3251.60 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 11:10:00 | 3257.50 | 3242.15 | 0.00 | ORB-long ORB[3212.00,3251.00] vol=1.7x ATR=5.42 |
| Stop hit — per-position SL triggered | 2026-04-21 11:15:00 | 3252.08 | 3242.40 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:10:00 | 3206.10 | 3220.68 | 0.00 | ORB-short ORB[3212.00,3249.90] vol=3.5x ATR=7.28 |
| Stop hit — per-position SL triggered | 2026-04-22 10:45:00 | 3213.38 | 3212.57 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:40:00 | 3141.50 | 3121.58 | 0.00 | ORB-long ORB[3090.10,3124.00] vol=1.8x ATR=7.90 |
| Stop hit — per-position SL triggered | 2026-04-28 09:50:00 | 3133.60 | 3123.63 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:45:00 | 3155.50 | 3129.69 | 0.00 | ORB-long ORB[3092.00,3138.40] vol=1.8x ATR=11.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 09:50:00 | 3172.08 | 3137.17 | 0.00 | T1 1.5R @ 3172.08 |
| Target hit | 2026-04-29 11:45:00 | 3162.00 | 3163.71 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — SELL (started 2026-04-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 09:55:00 | 3067.20 | 3084.08 | 0.00 | ORB-short ORB[3077.00,3116.90] vol=1.6x ATR=11.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 10:25:00 | 3050.01 | 3072.00 | 0.00 | T1 1.5R @ 3050.01 |
| Stop hit — per-position SL triggered | 2026-04-30 12:05:00 | 3067.20 | 3066.94 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-05-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:30:00 | 3365.90 | 3339.30 | 0.00 | ORB-long ORB[3305.00,3348.60] vol=2.1x ATR=11.89 |
| Stop hit — per-position SL triggered | 2026-05-07 10:40:00 | 3354.01 | 3354.79 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:30:00 | 3630.80 | 2026-02-10 09:40:00 | 3640.82 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2026-02-10 09:30:00 | 3630.80 | 2026-02-10 09:50:00 | 3630.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-11 09:45:00 | 3746.00 | 2026-02-11 10:20:00 | 3761.21 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-02-11 09:45:00 | 3746.00 | 2026-02-11 11:50:00 | 3746.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-17 11:15:00 | 3475.70 | 2026-02-17 11:55:00 | 3483.11 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-19 10:45:00 | 3489.00 | 2026-02-19 10:50:00 | 3496.08 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-02-25 11:00:00 | 3488.90 | 2026-02-25 11:50:00 | 3500.14 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2026-02-25 11:00:00 | 3488.90 | 2026-02-25 12:10:00 | 3488.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-11 10:30:00 | 3242.50 | 2026-03-11 10:35:00 | 3227.27 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-03-11 10:30:00 | 3242.50 | 2026-03-11 15:20:00 | 3175.70 | TARGET_HIT | 0.50 | 2.06% |
| BUY | retest1 | 2026-03-20 11:15:00 | 3104.10 | 2026-03-20 11:30:00 | 3092.99 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-04-13 09:35:00 | 3189.70 | 2026-04-13 09:40:00 | 3202.87 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-04-16 09:50:00 | 3245.20 | 2026-04-16 10:25:00 | 3230.86 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-04-16 09:50:00 | 3245.20 | 2026-04-16 10:45:00 | 3245.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 11:10:00 | 3257.50 | 2026-04-21 11:15:00 | 3252.08 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2026-04-22 10:10:00 | 3206.10 | 2026-04-22 10:45:00 | 3213.38 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-04-28 09:40:00 | 3141.50 | 2026-04-28 09:50:00 | 3133.60 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-04-29 09:45:00 | 3155.50 | 2026-04-29 09:50:00 | 3172.08 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-04-29 09:45:00 | 3155.50 | 2026-04-29 11:45:00 | 3162.00 | TARGET_HIT | 0.50 | 0.21% |
| SELL | retest1 | 2026-04-30 09:55:00 | 3067.20 | 2026-04-30 10:25:00 | 3050.01 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-04-30 09:55:00 | 3067.20 | 2026-04-30 12:05:00 | 3067.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-07 09:30:00 | 3365.90 | 2026-05-07 10:40:00 | 3354.01 | STOP_HIT | 1.00 | -0.35% |
