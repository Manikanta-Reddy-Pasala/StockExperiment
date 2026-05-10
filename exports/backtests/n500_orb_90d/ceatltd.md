# Ceat Ltd. (CEATLTD)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 3326.00
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
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 11
- **Target hits / Stop hits / Partials:** 3 / 11 / 7
- **Avg / median % per leg:** 0.22% / 0.00%
- **Sum % (uncompounded):** 4.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 6 | 42.9% | 2 | 8 | 4 | 0.19% | 2.7% |
| BUY @ 2nd Alert (retest1) | 14 | 6 | 42.9% | 2 | 8 | 4 | 0.19% | 2.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 4 | 57.1% | 1 | 3 | 3 | 0.29% | 2.0% |
| SELL @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 1 | 3 | 3 | 0.29% | 2.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 21 | 10 | 47.6% | 3 | 11 | 7 | 0.22% | 4.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:40:00 | 4003.30 | 3995.51 | 0.00 | ORB-long ORB[3928.00,3985.00] vol=1.8x ATR=30.09 |
| Target hit | 2026-02-09 15:20:00 | 4015.80 | 4006.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2026-02-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:40:00 | 4091.90 | 4063.12 | 0.00 | ORB-long ORB[4011.00,4071.00] vol=5.2x ATR=17.89 |
| Stop hit — per-position SL triggered | 2026-02-10 09:50:00 | 4074.01 | 4064.40 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:15:00 | 4026.40 | 4038.68 | 0.00 | ORB-short ORB[4028.80,4066.10] vol=4.4x ATR=9.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:30:00 | 4012.15 | 4036.82 | 0.00 | T1 1.5R @ 4012.15 |
| Target hit | 2026-02-12 15:20:00 | 3986.20 | 4012.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-02-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 10:40:00 | 3874.30 | 3922.81 | 0.00 | ORB-short ORB[3930.00,3964.00] vol=2.6x ATR=12.89 |
| Stop hit — per-position SL triggered | 2026-02-16 10:45:00 | 3887.19 | 3911.70 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 09:40:00 | 3825.90 | 3811.72 | 0.00 | ORB-long ORB[3771.80,3822.20] vol=1.6x ATR=12.09 |
| Stop hit — per-position SL triggered | 2026-02-23 09:55:00 | 3813.81 | 3812.93 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 3727.70 | 3750.46 | 0.00 | ORB-short ORB[3742.00,3781.40] vol=1.6x ATR=12.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:55:00 | 3708.24 | 3731.28 | 0.00 | T1 1.5R @ 3708.24 |
| Stop hit — per-position SL triggered | 2026-02-24 15:05:00 | 3727.70 | 3716.11 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:50:00 | 3807.90 | 3781.44 | 0.00 | ORB-long ORB[3752.40,3795.00] vol=1.7x ATR=14.43 |
| Stop hit — per-position SL triggered | 2026-02-25 10:55:00 | 3793.47 | 3782.82 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:35:00 | 3487.60 | 3453.88 | 0.00 | ORB-long ORB[3422.00,3465.00] vol=3.8x ATR=16.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 10:25:00 | 3512.02 | 3471.31 | 0.00 | T1 1.5R @ 3512.02 |
| Target hit | 2026-03-18 15:20:00 | 3550.80 | 3519.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2026-03-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 11:00:00 | 3548.20 | 3516.51 | 0.00 | ORB-long ORB[3501.00,3542.90] vol=3.1x ATR=14.70 |
| Stop hit — per-position SL triggered | 2026-03-20 12:15:00 | 3533.50 | 3525.07 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 09:45:00 | 3367.90 | 3337.10 | 0.00 | ORB-long ORB[3302.50,3345.50] vol=2.0x ATR=16.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-07 10:50:00 | 3392.34 | 3356.43 | 0.00 | T1 1.5R @ 3392.34 |
| Stop hit — per-position SL triggered | 2026-04-07 10:55:00 | 3367.90 | 3358.56 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:50:00 | 3638.90 | 3611.80 | 0.00 | ORB-long ORB[3550.00,3599.40] vol=2.1x ATR=12.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 12:05:00 | 3657.77 | 3622.67 | 0.00 | T1 1.5R @ 3657.77 |
| Stop hit — per-position SL triggered | 2026-04-10 12:25:00 | 3638.90 | 3624.67 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 11:10:00 | 3602.00 | 3557.81 | 0.00 | ORB-long ORB[3514.30,3561.70] vol=1.7x ATR=12.71 |
| Stop hit — per-position SL triggered | 2026-04-13 12:55:00 | 3589.29 | 3563.19 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:45:00 | 3634.80 | 3659.25 | 0.00 | ORB-short ORB[3647.10,3684.40] vol=1.9x ATR=11.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 09:50:00 | 3617.41 | 3656.15 | 0.00 | T1 1.5R @ 3617.41 |
| Stop hit — per-position SL triggered | 2026-04-16 10:45:00 | 3634.80 | 3641.72 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 11:05:00 | 3550.10 | 3521.05 | 0.00 | ORB-long ORB[3480.20,3526.60] vol=2.0x ATR=11.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 12:25:00 | 3567.89 | 3530.92 | 0.00 | T1 1.5R @ 3567.89 |
| Stop hit — per-position SL triggered | 2026-04-27 14:55:00 | 3550.10 | 3541.03 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:40:00 | 4003.30 | 2026-02-09 15:20:00 | 4015.80 | TARGET_HIT | 1.00 | 0.31% |
| BUY | retest1 | 2026-02-10 09:40:00 | 4091.90 | 2026-02-10 09:50:00 | 4074.01 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-02-12 11:15:00 | 4026.40 | 2026-02-12 11:30:00 | 4012.15 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-02-12 11:15:00 | 4026.40 | 2026-02-12 15:20:00 | 3986.20 | TARGET_HIT | 0.50 | 1.00% |
| SELL | retest1 | 2026-02-16 10:40:00 | 3874.30 | 2026-02-16 10:45:00 | 3887.19 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-23 09:40:00 | 3825.90 | 2026-02-23 09:55:00 | 3813.81 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-02-24 09:30:00 | 3727.70 | 2026-02-24 11:55:00 | 3708.24 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-02-24 09:30:00 | 3727.70 | 2026-02-24 15:05:00 | 3727.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 10:50:00 | 3807.90 | 2026-02-25 10:55:00 | 3793.47 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-03-18 09:35:00 | 3487.60 | 2026-03-18 10:25:00 | 3512.02 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-03-18 09:35:00 | 3487.60 | 2026-03-18 15:20:00 | 3550.80 | TARGET_HIT | 0.50 | 1.81% |
| BUY | retest1 | 2026-03-20 11:00:00 | 3548.20 | 2026-03-20 12:15:00 | 3533.50 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-07 09:45:00 | 3367.90 | 2026-04-07 10:50:00 | 3392.34 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2026-04-07 09:45:00 | 3367.90 | 2026-04-07 10:55:00 | 3367.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-10 10:50:00 | 3638.90 | 2026-04-10 12:05:00 | 3657.77 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-04-10 10:50:00 | 3638.90 | 2026-04-10 12:25:00 | 3638.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-13 11:10:00 | 3602.00 | 2026-04-13 12:55:00 | 3589.29 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-16 09:45:00 | 3634.80 | 2026-04-16 09:50:00 | 3617.41 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-04-16 09:45:00 | 3634.80 | 2026-04-16 10:45:00 | 3634.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 11:05:00 | 3550.10 | 2026-04-27 12:25:00 | 3567.89 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-04-27 11:05:00 | 3550.10 | 2026-04-27 14:55:00 | 3550.10 | STOP_HIT | 0.50 | 0.00% |
