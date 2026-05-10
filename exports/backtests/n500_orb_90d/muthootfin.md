# Muthoot Finance Ltd. (MUTHOOTFIN)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 3535.00
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
| ENTRY1 | 11 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 8
- **Target hits / Stop hits / Partials:** 3 / 8 / 4
- **Avg / median % per leg:** 0.30% / 0.00%
- **Sum % (uncompounded):** 4.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 3 | 37.5% | 1 | 5 | 2 | 0.31% | 2.5% |
| BUY @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 1 | 5 | 2 | 0.31% | 2.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 4 | 57.1% | 2 | 3 | 2 | 0.28% | 2.0% |
| SELL @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 2 | 3 | 2 | 0.28% | 2.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 15 | 7 | 46.7% | 3 | 8 | 4 | 0.30% | 4.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 11:10:00 | 3483.20 | 3465.09 | 0.00 | ORB-long ORB[3444.00,3482.80] vol=2.4x ATR=7.42 |
| Stop hit — per-position SL triggered | 2026-02-24 11:40:00 | 3475.78 | 3466.75 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-03-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 10:50:00 | 3334.10 | 3379.82 | 0.00 | ORB-short ORB[3377.00,3422.40] vol=1.6x ATR=12.36 |
| Stop hit — per-position SL triggered | 2026-03-04 11:05:00 | 3346.46 | 3376.61 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 11:10:00 | 3266.80 | 3291.85 | 0.00 | ORB-short ORB[3285.70,3326.90] vol=1.8x ATR=9.90 |
| Stop hit — per-position SL triggered | 2026-03-06 11:35:00 | 3276.70 | 3289.67 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:55:00 | 3213.90 | 3246.97 | 0.00 | ORB-short ORB[3260.60,3302.40] vol=1.5x ATR=8.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:20:00 | 3200.71 | 3236.44 | 0.00 | T1 1.5R @ 3200.71 |
| Target hit | 2026-03-11 15:20:00 | 3165.00 | 3202.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2026-03-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 09:45:00 | 3159.90 | 3146.07 | 0.00 | ORB-long ORB[3119.10,3156.50] vol=6.5x ATR=12.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 10:40:00 | 3178.20 | 3161.50 | 0.00 | T1 1.5R @ 3178.20 |
| Target hit | 2026-03-12 15:20:00 | 3244.90 | 3179.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 3483.20 | 3465.93 | 0.00 | ORB-long ORB[3436.20,3478.00] vol=2.4x ATR=12.83 |
| Stop hit — per-position SL triggered | 2026-03-18 09:35:00 | 3470.37 | 3467.15 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-06 10:55:00 | 3216.00 | 3211.99 | 0.00 | ORB-long ORB[3168.00,3207.20] vol=1.8x ATR=12.31 |
| Stop hit — per-position SL triggered | 2026-04-06 11:00:00 | 3203.69 | 3211.82 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 09:35:00 | 3631.00 | 3653.81 | 0.00 | ORB-short ORB[3633.60,3678.50] vol=2.3x ATR=16.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 09:45:00 | 3606.58 | 3645.34 | 0.00 | T1 1.5R @ 3606.58 |
| Target hit | 2026-04-15 10:55:00 | 3619.30 | 3618.00 | 0.00 | Trail-exit close>VWAP |

### Cycle 9 — BUY (started 2026-04-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:55:00 | 3595.50 | 3571.72 | 0.00 | ORB-long ORB[3528.00,3579.20] vol=1.7x ATR=10.62 |
| Stop hit — per-position SL triggered | 2026-04-21 10:10:00 | 3584.88 | 3579.63 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:30:00 | 3508.40 | 3496.99 | 0.00 | ORB-long ORB[3475.20,3506.10] vol=1.8x ATR=11.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:20:00 | 3525.19 | 3505.49 | 0.00 | T1 1.5R @ 3525.19 |
| Stop hit — per-position SL triggered | 2026-04-28 10:55:00 | 3508.40 | 3507.39 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-05-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:00:00 | 3470.10 | 3485.10 | 0.00 | ORB-short ORB[3480.00,3522.60] vol=1.5x ATR=9.95 |
| Stop hit — per-position SL triggered | 2026-05-06 10:40:00 | 3480.05 | 3479.65 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-24 11:10:00 | 3483.20 | 2026-02-24 11:40:00 | 3475.78 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-03-04 10:50:00 | 3334.10 | 2026-03-04 11:05:00 | 3346.46 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-03-06 11:10:00 | 3266.80 | 2026-03-06 11:35:00 | 3276.70 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-11 10:55:00 | 3213.90 | 2026-03-11 11:20:00 | 3200.71 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-03-11 10:55:00 | 3213.90 | 2026-03-11 15:20:00 | 3165.00 | TARGET_HIT | 0.50 | 1.52% |
| BUY | retest1 | 2026-03-12 09:45:00 | 3159.90 | 2026-03-12 10:40:00 | 3178.20 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-03-12 09:45:00 | 3159.90 | 2026-03-12 15:20:00 | 3244.90 | TARGET_HIT | 0.50 | 2.69% |
| BUY | retest1 | 2026-03-18 09:30:00 | 3483.20 | 2026-03-18 09:35:00 | 3470.37 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-06 10:55:00 | 3216.00 | 2026-04-06 11:00:00 | 3203.69 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-04-15 09:35:00 | 3631.00 | 2026-04-15 09:45:00 | 3606.58 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2026-04-15 09:35:00 | 3631.00 | 2026-04-15 10:55:00 | 3619.30 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2026-04-21 09:55:00 | 3595.50 | 2026-04-21 10:10:00 | 3584.88 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-28 09:30:00 | 3508.40 | 2026-04-28 10:20:00 | 3525.19 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-04-28 09:30:00 | 3508.40 | 2026-04-28 10:55:00 | 3508.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-06 10:00:00 | 3470.10 | 2026-05-06 10:40:00 | 3480.05 | STOP_HIT | 1.00 | -0.29% |
