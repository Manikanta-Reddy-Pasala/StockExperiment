# Thermax Ltd. (THERMAX)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 4707.00
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
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 14
- **Target hits / Stop hits / Partials:** 3 / 14 / 6
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 4.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 7 | 53.8% | 3 | 6 | 4 | 0.40% | 5.2% |
| BUY @ 2nd Alert (retest1) | 13 | 7 | 53.8% | 3 | 6 | 4 | 0.40% | 5.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 2 | 20.0% | 0 | 8 | 2 | -0.10% | -1.0% |
| SELL @ 2nd Alert (retest1) | 10 | 2 | 20.0% | 0 | 8 | 2 | -0.10% | -1.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 23 | 9 | 39.1% | 3 | 14 | 6 | 0.18% | 4.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 11:00:00 | 2949.40 | 2937.07 | 0.00 | ORB-long ORB[2901.30,2943.90] vol=2.0x ATR=6.94 |
| Stop hit — per-position SL triggered | 2026-02-10 11:25:00 | 2942.46 | 2938.20 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:05:00 | 2956.60 | 2937.70 | 0.00 | ORB-long ORB[2915.60,2948.50] vol=2.7x ATR=12.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:10:00 | 2974.67 | 2948.24 | 0.00 | T1 1.5R @ 2974.67 |
| Target hit | 2026-02-17 15:20:00 | 3056.00 | 3012.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-02-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:00:00 | 3042.00 | 3065.37 | 0.00 | ORB-short ORB[3048.00,3073.90] vol=1.7x ATR=11.42 |
| Stop hit — per-position SL triggered | 2026-02-18 10:15:00 | 3053.42 | 3061.37 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 3066.10 | 3074.97 | 0.00 | ORB-short ORB[3068.00,3100.00] vol=1.6x ATR=8.29 |
| Stop hit — per-position SL triggered | 2026-02-19 09:35:00 | 3074.39 | 3074.87 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:40:00 | 3027.80 | 3001.54 | 0.00 | ORB-long ORB[2975.00,3008.70] vol=2.0x ATR=10.80 |
| Stop hit — per-position SL triggered | 2026-02-20 10:55:00 | 3017.00 | 3003.12 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:30:00 | 3172.40 | 3153.39 | 0.00 | ORB-long ORB[3133.30,3170.00] vol=3.4x ATR=10.53 |
| Stop hit — per-position SL triggered | 2026-02-25 09:35:00 | 3161.87 | 3158.99 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-26 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:25:00 | 3180.30 | 3210.28 | 0.00 | ORB-short ORB[3202.50,3246.60] vol=1.9x ATR=11.13 |
| Stop hit — per-position SL triggered | 2026-02-26 10:50:00 | 3191.43 | 3206.33 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-02-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:50:00 | 3084.50 | 3107.04 | 0.00 | ORB-short ORB[3106.70,3148.00] vol=1.8x ATR=8.97 |
| Stop hit — per-position SL triggered | 2026-02-27 11:05:00 | 3093.47 | 3104.18 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:30:00 | 3028.90 | 3052.94 | 0.00 | ORB-short ORB[3045.90,3079.30] vol=2.1x ATR=11.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:40:00 | 3011.14 | 3041.71 | 0.00 | T1 1.5R @ 3011.14 |
| Stop hit — per-position SL triggered | 2026-03-04 09:45:00 | 3028.90 | 3040.17 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 10:00:00 | 3128.00 | 3112.98 | 0.00 | ORB-long ORB[3040.00,3086.60] vol=2.3x ATR=15.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:20:00 | 3151.98 | 3127.42 | 0.00 | T1 1.5R @ 3151.98 |
| Target hit | 2026-03-06 14:45:00 | 3155.60 | 3157.20 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — BUY (started 2026-03-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:30:00 | 3143.20 | 3115.81 | 0.00 | ORB-long ORB[3084.90,3125.70] vol=2.3x ATR=10.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 10:35:00 | 3159.29 | 3123.04 | 0.00 | T1 1.5R @ 3159.29 |
| Target hit | 2026-03-10 13:40:00 | 3160.00 | 3161.70 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — BUY (started 2026-04-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:45:00 | 3619.90 | 3577.79 | 0.00 | ORB-long ORB[3530.00,3583.90] vol=1.6x ATR=20.56 |
| Stop hit — per-position SL triggered | 2026-04-10 10:50:00 | 3599.34 | 3588.78 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:40:00 | 4121.60 | 4102.58 | 0.00 | ORB-long ORB[4085.20,4119.70] vol=1.7x ATR=20.48 |
| Stop hit — per-position SL triggered | 2026-04-17 09:45:00 | 4101.12 | 4107.15 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:15:00 | 4096.90 | 4071.78 | 0.00 | ORB-long ORB[4030.20,4084.90] vol=2.1x ATR=15.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 10:30:00 | 4119.66 | 4086.30 | 0.00 | T1 1.5R @ 4119.66 |
| Stop hit — per-position SL triggered | 2026-04-23 12:10:00 | 4096.90 | 4098.54 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:45:00 | 3994.80 | 4015.82 | 0.00 | ORB-short ORB[4022.10,4058.40] vol=2.0x ATR=11.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 11:45:00 | 3977.54 | 4004.74 | 0.00 | T1 1.5R @ 3977.54 |
| Stop hit — per-position SL triggered | 2026-04-29 12:55:00 | 3994.80 | 4001.64 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-04-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:40:00 | 3936.90 | 3964.82 | 0.00 | ORB-short ORB[3959.20,4004.00] vol=2.8x ATR=15.23 |
| Stop hit — per-position SL triggered | 2026-04-30 11:00:00 | 3952.13 | 3960.40 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-05-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:05:00 | 4147.10 | 4185.09 | 0.00 | ORB-short ORB[4172.00,4207.60] vol=1.6x ATR=14.89 |
| Stop hit — per-position SL triggered | 2026-05-05 11:30:00 | 4161.99 | 4181.62 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 11:00:00 | 2949.40 | 2026-02-10 11:25:00 | 2942.46 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-17 10:05:00 | 2956.60 | 2026-02-17 10:10:00 | 2974.67 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-02-17 10:05:00 | 2956.60 | 2026-02-17 15:20:00 | 3056.00 | TARGET_HIT | 0.50 | 3.36% |
| SELL | retest1 | 2026-02-18 10:00:00 | 3042.00 | 2026-02-18 10:15:00 | 3053.42 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-02-19 09:30:00 | 3066.10 | 2026-02-19 09:35:00 | 3074.39 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-20 10:40:00 | 3027.80 | 2026-02-20 10:55:00 | 3017.00 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-02-25 09:30:00 | 3172.40 | 2026-02-25 09:35:00 | 3161.87 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-26 10:25:00 | 3180.30 | 2026-02-26 10:50:00 | 3191.43 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-27 10:50:00 | 3084.50 | 2026-02-27 11:05:00 | 3093.47 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-04 09:30:00 | 3028.90 | 2026-03-04 09:40:00 | 3011.14 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-03-04 09:30:00 | 3028.90 | 2026-03-04 09:45:00 | 3028.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-06 10:00:00 | 3128.00 | 2026-03-06 10:20:00 | 3151.98 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2026-03-06 10:00:00 | 3128.00 | 2026-03-06 14:45:00 | 3155.60 | TARGET_HIT | 0.50 | 0.88% |
| BUY | retest1 | 2026-03-10 10:30:00 | 3143.20 | 2026-03-10 10:35:00 | 3159.29 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-03-10 10:30:00 | 3143.20 | 2026-03-10 13:40:00 | 3160.00 | TARGET_HIT | 0.50 | 0.53% |
| BUY | retest1 | 2026-04-10 09:45:00 | 3619.90 | 2026-04-10 10:50:00 | 3599.34 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2026-04-17 09:40:00 | 4121.60 | 2026-04-17 09:45:00 | 4101.12 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2026-04-23 10:15:00 | 4096.90 | 2026-04-23 10:30:00 | 4119.66 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-04-23 10:15:00 | 4096.90 | 2026-04-23 12:10:00 | 4096.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-29 10:45:00 | 3994.80 | 2026-04-29 11:45:00 | 3977.54 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-04-29 10:45:00 | 3994.80 | 2026-04-29 12:55:00 | 3994.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-30 10:40:00 | 3936.90 | 2026-04-30 11:00:00 | 3952.13 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-05-05 11:05:00 | 4147.10 | 2026-05-05 11:30:00 | 4161.99 | STOP_HIT | 1.00 | -0.36% |
