# Gujarat Fluorochemicals Ltd. (FLUOROCHEM)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 3777.60
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
| ENTRY1 | 14 |
| ENTRY2 | 0 |
| PARTIAL | 8 |
| TARGET_HIT | 3 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 11
- **Target hits / Stop hits / Partials:** 3 / 11 / 8
- **Avg / median % per leg:** 0.30% / 0.31%
- **Sum % (uncompounded):** 6.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 5 | 41.7% | 2 | 7 | 3 | 0.20% | 2.4% |
| BUY @ 2nd Alert (retest1) | 12 | 5 | 41.7% | 2 | 7 | 3 | 0.20% | 2.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 6 | 60.0% | 1 | 4 | 5 | 0.42% | 4.2% |
| SELL @ 2nd Alert (retest1) | 10 | 6 | 60.0% | 1 | 4 | 5 | 0.42% | 4.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 11 | 50.0% | 3 | 11 | 8 | 0.30% | 6.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:35:00 | 3444.00 | 3435.57 | 0.00 | ORB-long ORB[3406.90,3442.90] vol=7.7x ATR=18.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:00:00 | 3471.84 | 3441.73 | 0.00 | T1 1.5R @ 3471.84 |
| Target hit | 2026-02-09 15:20:00 | 3509.00 | 3481.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2026-02-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 09:45:00 | 3480.00 | 3464.02 | 0.00 | ORB-long ORB[3451.00,3470.00] vol=2.1x ATR=11.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 10:30:00 | 3496.71 | 3470.65 | 0.00 | T1 1.5R @ 3496.71 |
| Stop hit — per-position SL triggered | 2026-02-11 11:50:00 | 3480.00 | 3488.93 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:45:00 | 3363.60 | 3345.48 | 0.00 | ORB-long ORB[3291.30,3317.00] vol=2.8x ATR=9.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 11:15:00 | 3377.38 | 3350.62 | 0.00 | T1 1.5R @ 3377.38 |
| Target hit | 2026-02-17 15:20:00 | 3388.50 | 3372.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2026-02-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:35:00 | 3438.90 | 3416.68 | 0.00 | ORB-long ORB[3374.10,3399.40] vol=8.9x ATR=10.77 |
| Stop hit — per-position SL triggered | 2026-02-19 09:40:00 | 3428.13 | 3416.71 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:50:00 | 3430.10 | 3408.68 | 0.00 | ORB-long ORB[3381.00,3409.00] vol=7.6x ATR=8.46 |
| Stop hit — per-position SL triggered | 2026-02-20 13:25:00 | 3421.64 | 3422.51 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:55:00 | 3397.50 | 3416.95 | 0.00 | ORB-short ORB[3406.60,3440.00] vol=3.8x ATR=6.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:40:00 | 3387.06 | 3414.10 | 0.00 | T1 1.5R @ 3387.06 |
| Stop hit — per-position SL triggered | 2026-02-23 11:55:00 | 3397.50 | 3410.06 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:45:00 | 3432.30 | 3452.93 | 0.00 | ORB-short ORB[3445.00,3490.50] vol=4.5x ATR=9.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:30:00 | 3417.41 | 3445.91 | 0.00 | T1 1.5R @ 3417.41 |
| Stop hit — per-position SL triggered | 2026-02-26 13:05:00 | 3432.30 | 3443.71 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:25:00 | 3128.20 | 3153.31 | 0.00 | ORB-short ORB[3145.40,3174.00] vol=1.7x ATR=12.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 11:10:00 | 3109.97 | 3144.47 | 0.00 | T1 1.5R @ 3109.97 |
| Target hit | 2026-03-13 14:35:00 | 3076.00 | 3068.33 | 0.00 | Trail-exit close>VWAP |

### Cycle 9 — BUY (started 2026-03-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:45:00 | 3240.00 | 3229.67 | 0.00 | ORB-long ORB[3203.20,3231.00] vol=1.9x ATR=9.06 |
| Stop hit — per-position SL triggered | 2026-03-25 11:05:00 | 3230.94 | 3231.03 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 10:45:00 | 3256.00 | 3204.86 | 0.00 | ORB-long ORB[3161.00,3208.20] vol=3.8x ATR=13.71 |
| Stop hit — per-position SL triggered | 2026-04-07 10:50:00 | 3242.29 | 3219.23 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:35:00 | 3295.40 | 3283.62 | 0.00 | ORB-long ORB[3278.20,3292.10] vol=4.0x ATR=11.51 |
| Stop hit — per-position SL triggered | 2026-04-21 11:30:00 | 3283.89 | 3284.45 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:30:00 | 3300.00 | 3285.28 | 0.00 | ORB-long ORB[3265.00,3296.80] vol=2.4x ATR=10.69 |
| Stop hit — per-position SL triggered | 2026-04-23 09:40:00 | 3289.31 | 3286.99 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-05-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 11:10:00 | 3813.80 | 3843.77 | 0.00 | ORB-short ORB[3834.80,3878.00] vol=4.7x ATR=14.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 11:35:00 | 3791.71 | 3840.46 | 0.00 | T1 1.5R @ 3791.71 |
| Stop hit — per-position SL triggered | 2026-05-07 11:55:00 | 3813.80 | 3839.11 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-05-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:35:00 | 3783.00 | 3799.51 | 0.00 | ORB-short ORB[3785.00,3826.20] vol=3.8x ATR=16.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 10:15:00 | 3758.76 | 3786.60 | 0.00 | T1 1.5R @ 3758.76 |
| Stop hit — per-position SL triggered | 2026-05-08 10:45:00 | 3783.00 | 3782.73 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:35:00 | 3444.00 | 2026-02-09 11:00:00 | 3471.84 | PARTIAL | 0.50 | 0.81% |
| BUY | retest1 | 2026-02-09 10:35:00 | 3444.00 | 2026-02-09 15:20:00 | 3509.00 | TARGET_HIT | 0.50 | 1.89% |
| BUY | retest1 | 2026-02-11 09:45:00 | 3480.00 | 2026-02-11 10:30:00 | 3496.71 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-02-11 09:45:00 | 3480.00 | 2026-02-11 11:50:00 | 3480.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 10:45:00 | 3363.60 | 2026-02-17 11:15:00 | 3377.38 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-02-17 10:45:00 | 3363.60 | 2026-02-17 15:20:00 | 3388.50 | TARGET_HIT | 0.50 | 0.74% |
| BUY | retest1 | 2026-02-19 09:35:00 | 3438.90 | 2026-02-19 09:40:00 | 3428.13 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-20 10:50:00 | 3430.10 | 2026-02-20 13:25:00 | 3421.64 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-02-23 10:55:00 | 3397.50 | 2026-02-23 11:40:00 | 3387.06 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-02-23 10:55:00 | 3397.50 | 2026-02-23 11:55:00 | 3397.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-26 10:45:00 | 3432.30 | 2026-02-26 11:30:00 | 3417.41 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-02-26 10:45:00 | 3432.30 | 2026-02-26 13:05:00 | 3432.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-13 10:25:00 | 3128.20 | 2026-03-13 11:10:00 | 3109.97 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-03-13 10:25:00 | 3128.20 | 2026-03-13 14:35:00 | 3076.00 | TARGET_HIT | 0.50 | 1.67% |
| BUY | retest1 | 2026-03-25 10:45:00 | 3240.00 | 2026-03-25 11:05:00 | 3230.94 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-07 10:45:00 | 3256.00 | 2026-04-07 10:50:00 | 3242.29 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-04-21 10:35:00 | 3295.40 | 2026-04-21 11:30:00 | 3283.89 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-23 09:30:00 | 3300.00 | 2026-04-23 09:40:00 | 3289.31 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-05-07 11:10:00 | 3813.80 | 2026-05-07 11:35:00 | 3791.71 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-05-07 11:10:00 | 3813.80 | 2026-05-07 11:55:00 | 3813.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-08 09:35:00 | 3783.00 | 2026-05-08 10:15:00 | 3758.76 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2026-05-08 09:35:00 | 3783.00 | 2026-05-08 10:45:00 | 3783.00 | STOP_HIT | 0.50 | 0.00% |
