# Anand Rathi Wealth Ltd. (ANANDRATHI)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 3602.30
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
| PARTIAL | 8 |
| TARGET_HIT | 3 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 14
- **Target hits / Stop hits / Partials:** 3 / 14 / 8
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 3.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.20% | 2.2% |
| BUY @ 2nd Alert (retest1) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.20% | 2.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 7 | 50.0% | 2 | 7 | 5 | 0.09% | 1.3% |
| SELL @ 2nd Alert (retest1) | 14 | 7 | 50.0% | 2 | 7 | 5 | 0.09% | 1.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 25 | 11 | 44.0% | 3 | 14 | 8 | 0.14% | 3.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:45:00 | 2982.00 | 2987.64 | 0.00 | ORB-short ORB[2985.30,3019.80] vol=8.3x ATR=7.01 |
| Stop hit — per-position SL triggered | 2026-02-12 11:05:00 | 2989.01 | 2986.99 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:15:00 | 3028.00 | 3005.57 | 0.00 | ORB-long ORB[2983.40,3022.20] vol=3.6x ATR=9.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 10:45:00 | 3042.41 | 3020.02 | 0.00 | T1 1.5R @ 3042.41 |
| Stop hit — per-position SL triggered | 2026-02-16 10:50:00 | 3028.00 | 3020.75 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:40:00 | 3027.30 | 3023.61 | 0.00 | ORB-long ORB[3008.50,3022.00] vol=6.8x ATR=7.27 |
| Stop hit — per-position SL triggered | 2026-02-19 10:00:00 | 3020.03 | 3024.94 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:35:00 | 3015.10 | 3027.41 | 0.00 | ORB-short ORB[3017.40,3045.00] vol=1.7x ATR=7.11 |
| Stop hit — per-position SL triggered | 2026-02-23 10:55:00 | 3022.21 | 3026.41 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:45:00 | 3038.50 | 3048.51 | 0.00 | ORB-short ORB[3039.90,3063.30] vol=3.0x ATR=7.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 10:50:00 | 3027.17 | 3045.46 | 0.00 | T1 1.5R @ 3027.17 |
| Stop hit — per-position SL triggered | 2026-02-24 11:00:00 | 3038.50 | 3043.56 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 11:00:00 | 3068.20 | 3052.65 | 0.00 | ORB-long ORB[3041.10,3064.90] vol=2.3x ATR=6.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:05:00 | 3077.40 | 3056.49 | 0.00 | T1 1.5R @ 3077.40 |
| Stop hit — per-position SL triggered | 2026-02-25 11:10:00 | 3068.20 | 3058.14 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:40:00 | 3069.00 | 3064.61 | 0.00 | ORB-long ORB[3045.00,3065.00] vol=1.9x ATR=6.25 |
| Stop hit — per-position SL triggered | 2026-02-26 10:50:00 | 3062.75 | 3064.69 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-02-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:50:00 | 3051.40 | 3057.60 | 0.00 | ORB-short ORB[3060.00,3078.50] vol=1.5x ATR=8.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:55:00 | 3038.56 | 3052.72 | 0.00 | T1 1.5R @ 3038.56 |
| Stop hit — per-position SL triggered | 2026-02-27 10:00:00 | 3051.40 | 3052.58 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-02 10:55:00 | 3082.00 | 3046.77 | 0.00 | ORB-long ORB[3000.00,3044.20] vol=2.0x ATR=10.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 11:20:00 | 3098.18 | 3055.76 | 0.00 | T1 1.5R @ 3098.18 |
| Target hit | 2026-03-02 15:20:00 | 3152.60 | 3124.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2026-03-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 10:35:00 | 3049.00 | 3055.19 | 0.00 | ORB-short ORB[3060.10,3084.10] vol=2.2x ATR=9.76 |
| Stop hit — per-position SL triggered | 2026-03-17 10:55:00 | 3058.76 | 3053.29 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 09:30:00 | 3016.00 | 3032.25 | 0.00 | ORB-short ORB[3028.60,3053.20] vol=2.5x ATR=12.29 |
| Stop hit — per-position SL triggered | 2026-03-19 09:55:00 | 3028.29 | 3026.35 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:50:00 | 3009.80 | 2993.00 | 0.00 | ORB-long ORB[2970.60,2990.00] vol=1.7x ATR=8.10 |
| Stop hit — per-position SL triggered | 2026-03-25 10:55:00 | 3001.70 | 2993.20 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:35:00 | 3666.60 | 3639.81 | 0.00 | ORB-long ORB[3614.00,3649.00] vol=3.0x ATR=13.65 |
| Stop hit — per-position SL triggered | 2026-04-15 10:00:00 | 3652.95 | 3650.22 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 09:30:00 | 3648.30 | 3655.18 | 0.00 | ORB-short ORB[3650.00,3663.90] vol=1.8x ATR=9.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:40:00 | 3634.13 | 3649.21 | 0.00 | T1 1.5R @ 3634.13 |
| Target hit | 2026-04-21 10:10:00 | 3646.00 | 3642.04 | 0.00 | Trail-exit close>VWAP |

### Cycle 15 — SELL (started 2026-04-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:30:00 | 3592.90 | 3602.64 | 0.00 | ORB-short ORB[3607.60,3640.20] vol=1.8x ATR=10.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:20:00 | 3577.64 | 3597.00 | 0.00 | T1 1.5R @ 3577.64 |
| Target hit | 2026-04-24 12:15:00 | 3575.80 | 3572.35 | 0.00 | Trail-exit close>VWAP |

### Cycle 16 — SELL (started 2026-04-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 10:55:00 | 3571.10 | 3583.87 | 0.00 | ORB-short ORB[3578.10,3609.80] vol=2.5x ATR=8.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 11:20:00 | 3558.75 | 3581.08 | 0.00 | T1 1.5R @ 3558.75 |
| Stop hit — per-position SL triggered | 2026-04-27 11:30:00 | 3571.10 | 3580.59 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 09:50:00 | 3625.00 | 3607.44 | 0.00 | ORB-long ORB[3588.10,3618.20] vol=1.6x ATR=10.48 |
| Stop hit — per-position SL triggered | 2026-04-30 10:10:00 | 3614.52 | 3616.62 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-12 10:45:00 | 2982.00 | 2026-02-12 11:05:00 | 2989.01 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-16 10:15:00 | 3028.00 | 2026-02-16 10:45:00 | 3042.41 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-02-16 10:15:00 | 3028.00 | 2026-02-16 10:50:00 | 3028.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-19 09:40:00 | 3027.30 | 2026-02-19 10:00:00 | 3020.03 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-23 10:35:00 | 3015.10 | 2026-02-23 10:55:00 | 3022.21 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-24 10:45:00 | 3038.50 | 2026-02-24 10:50:00 | 3027.17 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-02-24 10:45:00 | 3038.50 | 2026-02-24 11:00:00 | 3038.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 11:00:00 | 3068.20 | 2026-02-25 11:05:00 | 3077.40 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-02-25 11:00:00 | 3068.20 | 2026-02-25 11:10:00 | 3068.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-26 10:40:00 | 3069.00 | 2026-02-26 10:50:00 | 3062.75 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-27 09:50:00 | 3051.40 | 2026-02-27 09:55:00 | 3038.56 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-02-27 09:50:00 | 3051.40 | 2026-02-27 10:00:00 | 3051.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-02 10:55:00 | 3082.00 | 2026-03-02 11:20:00 | 3098.18 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-03-02 10:55:00 | 3082.00 | 2026-03-02 15:20:00 | 3152.60 | TARGET_HIT | 0.50 | 2.29% |
| SELL | retest1 | 2026-03-17 10:35:00 | 3049.00 | 2026-03-17 10:55:00 | 3058.76 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-03-19 09:30:00 | 3016.00 | 2026-03-19 09:55:00 | 3028.29 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-03-25 10:50:00 | 3009.80 | 2026-03-25 10:55:00 | 3001.70 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-04-15 09:35:00 | 3666.60 | 2026-04-15 10:00:00 | 3652.95 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-04-21 09:30:00 | 3648.30 | 2026-04-21 09:40:00 | 3634.13 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-04-21 09:30:00 | 3648.30 | 2026-04-21 10:10:00 | 3646.00 | TARGET_HIT | 0.50 | 0.06% |
| SELL | retest1 | 2026-04-24 10:30:00 | 3592.90 | 2026-04-24 11:20:00 | 3577.64 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-04-24 10:30:00 | 3592.90 | 2026-04-24 12:15:00 | 3575.80 | TARGET_HIT | 0.50 | 0.48% |
| SELL | retest1 | 2026-04-27 10:55:00 | 3571.10 | 2026-04-27 11:20:00 | 3558.75 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-04-27 10:55:00 | 3571.10 | 2026-04-27 11:30:00 | 3571.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-30 09:50:00 | 3625.00 | 2026-04-30 10:10:00 | 3614.52 | STOP_HIT | 1.00 | -0.29% |
