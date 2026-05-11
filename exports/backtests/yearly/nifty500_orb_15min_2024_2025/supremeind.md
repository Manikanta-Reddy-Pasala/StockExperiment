# Supreme Industries Ltd. (SUPREMEIND)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 3654.00
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
| ENTRY1 | 38 |
| ENTRY2 | 0 |
| PARTIAL | 11 |
| TARGET_HIT | 4 |
| STOP_HIT | 34 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 49 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 34
- **Target hits / Stop hits / Partials:** 4 / 34 / 11
- **Avg / median % per leg:** -0.04% / -0.24%
- **Sum % (uncompounded):** -1.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 6 | 24.0% | 1 | 19 | 5 | -0.12% | -2.9% |
| BUY @ 2nd Alert (retest1) | 25 | 6 | 24.0% | 1 | 19 | 5 | -0.12% | -2.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 24 | 9 | 37.5% | 3 | 15 | 6 | 0.04% | 1.1% |
| SELL @ 2nd Alert (retest1) | 24 | 9 | 37.5% | 3 | 15 | 6 | 0.04% | 1.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 49 | 15 | 30.6% | 4 | 34 | 11 | -0.04% | -1.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 11:15:00 | 6039.00 | 6061.17 | 0.00 | ORB-short ORB[6040.05,6109.95] vol=2.1x ATR=13.89 |
| Stop hit — per-position SL triggered | 2024-07-05 13:45:00 | 6052.89 | 6056.67 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-08-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-19 10:25:00 | 5228.05 | 5243.58 | 0.00 | ORB-short ORB[5241.05,5307.35] vol=3.9x ATR=15.39 |
| Stop hit — per-position SL triggered | 2024-08-19 10:35:00 | 5243.44 | 5243.10 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-08-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 11:05:00 | 5380.00 | 5405.23 | 0.00 | ORB-short ORB[5385.00,5457.10] vol=4.1x ATR=13.14 |
| Stop hit — per-position SL triggered | 2024-08-27 11:20:00 | 5393.14 | 5403.59 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 5334.35 | 5392.45 | 0.00 | ORB-short ORB[5361.00,5434.80] vol=1.8x ATR=21.37 |
| Stop hit — per-position SL triggered | 2024-08-28 09:40:00 | 5355.72 | 5383.47 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-08-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 09:55:00 | 5399.00 | 5381.97 | 0.00 | ORB-long ORB[5322.70,5397.75] vol=2.4x ATR=17.88 |
| Stop hit — per-position SL triggered | 2024-08-29 10:05:00 | 5381.12 | 5384.23 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-09-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 10:10:00 | 5390.00 | 5381.00 | 0.00 | ORB-long ORB[5317.00,5385.00] vol=4.7x ATR=15.40 |
| Stop hit — per-position SL triggered | 2024-09-17 12:05:00 | 5374.60 | 5387.50 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-09-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 10:10:00 | 5400.55 | 5407.65 | 0.00 | ORB-short ORB[5402.00,5447.55] vol=3.0x ATR=10.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 12:20:00 | 5385.29 | 5402.95 | 0.00 | T1 1.5R @ 5385.29 |
| Target hit | 2024-09-24 15:00:00 | 5398.10 | 5397.94 | 0.00 | Trail-exit close>VWAP |

### Cycle 8 — SELL (started 2024-10-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 10:50:00 | 5229.90 | 5271.19 | 0.00 | ORB-short ORB[5260.00,5331.00] vol=2.9x ATR=13.98 |
| Stop hit — per-position SL triggered | 2024-10-16 11:35:00 | 5243.88 | 5262.04 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-10-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 11:00:00 | 5203.40 | 5211.67 | 0.00 | ORB-short ORB[5224.85,5271.10] vol=3.6x ATR=14.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:40:00 | 5181.87 | 5208.24 | 0.00 | T1 1.5R @ 5181.87 |
| Stop hit — per-position SL triggered | 2024-10-17 13:20:00 | 5203.40 | 5194.00 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 11:15:00 | 4519.20 | 4559.47 | 0.00 | ORB-short ORB[4580.00,4624.95] vol=1.7x ATR=20.98 |
| Stop hit — per-position SL triggered | 2024-11-13 12:20:00 | 4540.18 | 4539.80 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-12-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 09:45:00 | 4766.20 | 4780.05 | 0.00 | ORB-short ORB[4768.10,4810.00] vol=1.6x ATR=10.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 10:25:00 | 4749.92 | 4776.95 | 0.00 | T1 1.5R @ 4749.92 |
| Stop hit — per-position SL triggered | 2024-12-04 10:45:00 | 4766.20 | 4774.58 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-12-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 10:40:00 | 4926.60 | 4793.97 | 0.00 | ORB-long ORB[4738.55,4775.05] vol=2.3x ATR=21.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 10:50:00 | 4958.23 | 4816.58 | 0.00 | T1 1.5R @ 4958.23 |
| Stop hit — per-position SL triggered | 2024-12-09 11:05:00 | 4926.60 | 4827.66 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-12-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 11:00:00 | 4760.00 | 4781.09 | 0.00 | ORB-short ORB[4802.00,4840.85] vol=1.7x ATR=13.42 |
| Stop hit — per-position SL triggered | 2024-12-13 11:10:00 | 4773.42 | 4779.42 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-12-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 09:35:00 | 4804.00 | 4819.21 | 0.00 | ORB-short ORB[4815.00,4848.65] vol=2.0x ATR=13.87 |
| Stop hit — per-position SL triggered | 2024-12-16 09:40:00 | 4817.87 | 4820.96 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-12-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 10:40:00 | 5076.30 | 5026.66 | 0.00 | ORB-long ORB[4969.50,5012.20] vol=1.6x ATR=17.38 |
| Stop hit — per-position SL triggered | 2024-12-18 11:05:00 | 5058.92 | 5031.19 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-12-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-23 10:50:00 | 4804.00 | 4806.62 | 0.00 | ORB-short ORB[4811.55,4861.45] vol=4.8x ATR=15.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 13:05:00 | 4780.06 | 4801.29 | 0.00 | T1 1.5R @ 4780.06 |
| Target hit | 2024-12-23 14:15:00 | 4787.95 | 4786.05 | 0.00 | Trail-exit close>VWAP |

### Cycle 17 — SELL (started 2024-12-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-24 09:30:00 | 4722.00 | 4753.24 | 0.00 | ORB-short ORB[4744.40,4803.25] vol=1.5x ATR=16.11 |
| Stop hit — per-position SL triggered | 2024-12-24 09:55:00 | 4738.11 | 4737.12 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 11:15:00 | 4655.50 | 4692.76 | 0.00 | ORB-short ORB[4673.70,4735.95] vol=2.6x ATR=10.48 |
| Stop hit — per-position SL triggered | 2025-01-02 11:50:00 | 4665.98 | 4686.01 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-01-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 09:35:00 | 4578.90 | 4573.43 | 0.00 | ORB-long ORB[4531.30,4573.45] vol=3.1x ATR=19.79 |
| Stop hit — per-position SL triggered | 2025-01-07 10:00:00 | 4559.11 | 4571.93 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-01-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 10:05:00 | 4471.75 | 4483.70 | 0.00 | ORB-short ORB[4479.75,4545.00] vol=2.0x ATR=13.34 |
| Stop hit — per-position SL triggered | 2025-01-08 10:15:00 | 4485.09 | 4483.29 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-01-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 09:55:00 | 4347.30 | 4313.15 | 0.00 | ORB-long ORB[4274.00,4321.90] vol=1.6x ATR=17.67 |
| Stop hit — per-position SL triggered | 2025-01-17 10:30:00 | 4329.63 | 4317.72 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-01-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-21 10:10:00 | 4046.95 | 4009.38 | 0.00 | ORB-long ORB[3955.55,4015.00] vol=1.8x ATR=28.32 |
| Stop hit — per-position SL triggered | 2025-01-21 11:25:00 | 4018.63 | 4016.41 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-01-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 09:55:00 | 3985.30 | 3957.64 | 0.00 | ORB-long ORB[3876.10,3931.95] vol=2.0x ATR=18.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 10:00:00 | 4012.31 | 3969.95 | 0.00 | T1 1.5R @ 4012.31 |
| Target hit | 2025-01-23 11:35:00 | 4000.85 | 4006.69 | 0.00 | Trail-exit close<VWAP |

### Cycle 24 — SELL (started 2025-01-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 09:55:00 | 3921.80 | 3958.64 | 0.00 | ORB-short ORB[3969.05,4002.20] vol=1.5x ATR=14.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 10:15:00 | 3900.16 | 3946.72 | 0.00 | T1 1.5R @ 3900.16 |
| Stop hit — per-position SL triggered | 2025-01-24 10:45:00 | 3921.80 | 3937.98 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-02-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 09:50:00 | 4047.00 | 4020.46 | 0.00 | ORB-long ORB[3948.05,4007.95] vol=1.6x ATR=20.16 |
| Stop hit — per-position SL triggered | 2025-02-05 12:20:00 | 4026.84 | 4037.57 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-02-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-11 09:40:00 | 3960.20 | 3946.14 | 0.00 | ORB-long ORB[3912.90,3957.95] vol=2.4x ATR=13.28 |
| Stop hit — per-position SL triggered | 2025-02-11 09:45:00 | 3946.92 | 3946.84 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-03-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-06 10:40:00 | 3469.00 | 3423.72 | 0.00 | ORB-long ORB[3400.00,3437.00] vol=1.9x ATR=15.94 |
| Stop hit — per-position SL triggered | 2025-03-06 10:45:00 | 3453.06 | 3426.80 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-03-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 10:00:00 | 3355.50 | 3330.62 | 0.00 | ORB-long ORB[3285.00,3335.25] vol=2.3x ATR=17.65 |
| Stop hit — per-position SL triggered | 2025-03-11 10:20:00 | 3337.85 | 3342.38 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-03-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-13 11:10:00 | 3365.40 | 3351.26 | 0.00 | ORB-long ORB[3328.30,3363.45] vol=1.5x ATR=9.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-13 12:25:00 | 3379.22 | 3356.35 | 0.00 | T1 1.5R @ 3379.22 |
| Stop hit — per-position SL triggered | 2025-03-13 13:15:00 | 3365.40 | 3360.08 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-03-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:25:00 | 3402.30 | 3369.48 | 0.00 | ORB-long ORB[3346.75,3375.00] vol=3.4x ATR=10.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 10:40:00 | 3417.73 | 3375.83 | 0.00 | T1 1.5R @ 3417.73 |
| Stop hit — per-position SL triggered | 2025-03-18 10:55:00 | 3402.30 | 3380.30 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-03-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 10:30:00 | 3463.55 | 3444.12 | 0.00 | ORB-long ORB[3419.65,3448.80] vol=1.8x ATR=11.69 |
| Stop hit — per-position SL triggered | 2025-03-19 11:25:00 | 3451.86 | 3448.48 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-03-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 09:35:00 | 3485.00 | 3465.90 | 0.00 | ORB-long ORB[3430.25,3473.40] vol=2.9x ATR=14.40 |
| Stop hit — per-position SL triggered | 2025-03-20 10:05:00 | 3470.60 | 3469.18 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-03-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:35:00 | 3532.45 | 3512.36 | 0.00 | ORB-long ORB[3478.95,3514.40] vol=2.4x ATR=11.36 |
| Stop hit — per-position SL triggered | 2025-03-21 09:50:00 | 3521.09 | 3521.71 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-03-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 11:00:00 | 3482.05 | 3464.95 | 0.00 | ORB-long ORB[3431.00,3469.15] vol=3.3x ATR=9.85 |
| Stop hit — per-position SL triggered | 2025-03-26 11:20:00 | 3472.20 | 3467.18 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-04-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-01 11:00:00 | 3388.00 | 3426.78 | 0.00 | ORB-short ORB[3404.50,3451.85] vol=1.6x ATR=13.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-01 11:50:00 | 3368.07 | 3413.98 | 0.00 | T1 1.5R @ 3368.07 |
| Target hit | 2025-04-01 15:20:00 | 3329.10 | 3368.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — BUY (started 2025-04-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-11 09:35:00 | 3180.55 | 3168.33 | 0.00 | ORB-long ORB[3150.75,3179.75] vol=2.0x ATR=12.15 |
| Stop hit — per-position SL triggered | 2025-04-11 09:45:00 | 3168.40 | 3169.06 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-04-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-24 09:45:00 | 3448.50 | 3483.95 | 0.00 | ORB-short ORB[3476.30,3526.40] vol=3.2x ATR=13.63 |
| Stop hit — per-position SL triggered | 2025-04-24 09:50:00 | 3462.13 | 3482.13 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-05-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 10:00:00 | 3498.50 | 3466.53 | 0.00 | ORB-long ORB[3438.70,3490.30] vol=1.6x ATR=12.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 10:20:00 | 3517.70 | 3483.95 | 0.00 | T1 1.5R @ 3517.70 |
| Stop hit — per-position SL triggered | 2025-05-08 11:00:00 | 3498.50 | 3492.26 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-07-05 11:15:00 | 6039.00 | 2024-07-05 13:45:00 | 6052.89 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-08-19 10:25:00 | 5228.05 | 2024-08-19 10:35:00 | 5243.44 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-08-27 11:05:00 | 5380.00 | 2024-08-27 11:20:00 | 5393.14 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-08-28 09:30:00 | 5334.35 | 2024-08-28 09:40:00 | 5355.72 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-08-29 09:55:00 | 5399.00 | 2024-08-29 10:05:00 | 5381.12 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-09-17 10:10:00 | 5390.00 | 2024-09-17 12:05:00 | 5374.60 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-09-24 10:10:00 | 5400.55 | 2024-09-24 12:20:00 | 5385.29 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2024-09-24 10:10:00 | 5400.55 | 2024-09-24 15:00:00 | 5398.10 | TARGET_HIT | 0.50 | 0.05% |
| SELL | retest1 | 2024-10-16 10:50:00 | 5229.90 | 2024-10-16 11:35:00 | 5243.88 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-10-17 11:00:00 | 5203.40 | 2024-10-17 11:40:00 | 5181.87 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-10-17 11:00:00 | 5203.40 | 2024-10-17 13:20:00 | 5203.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-13 11:15:00 | 4519.20 | 2024-11-13 12:20:00 | 4540.18 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-12-04 09:45:00 | 4766.20 | 2024-12-04 10:25:00 | 4749.92 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-12-04 09:45:00 | 4766.20 | 2024-12-04 10:45:00 | 4766.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-09 10:40:00 | 4926.60 | 2024-12-09 10:50:00 | 4958.23 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-12-09 10:40:00 | 4926.60 | 2024-12-09 11:05:00 | 4926.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-13 11:00:00 | 4760.00 | 2024-12-13 11:10:00 | 4773.42 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-12-16 09:35:00 | 4804.00 | 2024-12-16 09:40:00 | 4817.87 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-12-18 10:40:00 | 5076.30 | 2024-12-18 11:05:00 | 5058.92 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-12-23 10:50:00 | 4804.00 | 2024-12-23 13:05:00 | 4780.06 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-12-23 10:50:00 | 4804.00 | 2024-12-23 14:15:00 | 4787.95 | TARGET_HIT | 0.50 | 0.33% |
| SELL | retest1 | 2024-12-24 09:30:00 | 4722.00 | 2024-12-24 09:55:00 | 4738.11 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-01-02 11:15:00 | 4655.50 | 2025-01-02 11:50:00 | 4665.98 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-01-07 09:35:00 | 4578.90 | 2025-01-07 10:00:00 | 4559.11 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-01-08 10:05:00 | 4471.75 | 2025-01-08 10:15:00 | 4485.09 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-01-17 09:55:00 | 4347.30 | 2025-01-17 10:30:00 | 4329.63 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-01-21 10:10:00 | 4046.95 | 2025-01-21 11:25:00 | 4018.63 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest1 | 2025-01-23 09:55:00 | 3985.30 | 2025-01-23 10:00:00 | 4012.31 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2025-01-23 09:55:00 | 3985.30 | 2025-01-23 11:35:00 | 4000.85 | TARGET_HIT | 0.50 | 0.39% |
| SELL | retest1 | 2025-01-24 09:55:00 | 3921.80 | 2025-01-24 10:15:00 | 3900.16 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-01-24 09:55:00 | 3921.80 | 2025-01-24 10:45:00 | 3921.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-05 09:50:00 | 4047.00 | 2025-02-05 12:20:00 | 4026.84 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-02-11 09:40:00 | 3960.20 | 2025-02-11 09:45:00 | 3946.92 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-03-06 10:40:00 | 3469.00 | 2025-03-06 10:45:00 | 3453.06 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-03-11 10:00:00 | 3355.50 | 2025-03-11 10:20:00 | 3337.85 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2025-03-13 11:10:00 | 3365.40 | 2025-03-13 12:25:00 | 3379.22 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-03-13 11:10:00 | 3365.40 | 2025-03-13 13:15:00 | 3365.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-18 10:25:00 | 3402.30 | 2025-03-18 10:40:00 | 3417.73 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-03-18 10:25:00 | 3402.30 | 2025-03-18 10:55:00 | 3402.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-19 10:30:00 | 3463.55 | 2025-03-19 11:25:00 | 3451.86 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-03-20 09:35:00 | 3485.00 | 2025-03-20 10:05:00 | 3470.60 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-03-21 09:35:00 | 3532.45 | 2025-03-21 09:50:00 | 3521.09 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-03-26 11:00:00 | 3482.05 | 2025-03-26 11:20:00 | 3472.20 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-04-01 11:00:00 | 3388.00 | 2025-04-01 11:50:00 | 3368.07 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2025-04-01 11:00:00 | 3388.00 | 2025-04-01 15:20:00 | 3329.10 | TARGET_HIT | 0.50 | 1.74% |
| BUY | retest1 | 2025-04-11 09:35:00 | 3180.55 | 2025-04-11 09:45:00 | 3168.40 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-04-24 09:45:00 | 3448.50 | 2025-04-24 09:50:00 | 3462.13 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-05-08 10:00:00 | 3498.50 | 2025-05-08 10:20:00 | 3517.70 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-05-08 10:00:00 | 3498.50 | 2025-05-08 11:00:00 | 3498.50 | STOP_HIT | 0.50 | 0.00% |
