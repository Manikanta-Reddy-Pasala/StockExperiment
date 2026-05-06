# HEROMOTOCO (HEROMOTOCO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:15:00 (4996 bars)
- **Last close:** 5170.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 3 |
| ALERT3 | 8 |
| PENDING | 33 |
| PENDING_CANCEL | 8 |
| ENTRY1 | 0 |
| ENTRY2 | 25 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 3 / 22
- **Target hits / Stop hits / Partials:** 1 / 22 / 2
- **Avg / median % per leg:** 1.04% / -0.98%
- **Sum % (uncompounded):** 26.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 1 | 5 | 1 | 5.66% | 39.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 2 | 28.6% | 1 | 5 | 1 | 5.66% | 39.6% |
| SELL (all) | 18 | 1 | 5.6% | 0 | 17 | 1 | -0.75% | -13.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 1 | 5.6% | 0 | 17 | 1 | -0.75% | -13.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 25 | 3 | 12.0% | 1 | 22 | 2 | 1.04% | 26.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 15:15:00 | 2920.00 | 2986.50 | 2986.62 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-09-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 14:15:00 | 2997.90 | 2984.67 | 2984.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 09:15:00 | 3081.95 | 2985.77 | 2985.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-21 12:15:00 | 2999.00 | 3004.97 | 2995.47 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 12:15:00 | 2999.00 | 3004.97 | 2995.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 12:15:00 | 2999.00 | 3004.97 | 2995.47 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2023-09-22 09:15:00 | 3013.80 | 3004.88 | 2995.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-22 10:15:00 | 3032.80 | 3005.16 | 2995.80 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-09-25 09:15:00 | 2994.25 | 3006.05 | 2996.54 | SL hit qty=1.00 sl=2994.25 alert=retest2 |
| Cross detected — sustain check pending | 2023-09-26 10:15:00 | 3017.20 | 3004.85 | 2996.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-09-26 11:15:00 | 3002.80 | 3004.82 | 2996.32 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-09-26 14:15:00 | 3036.30 | 3005.14 | 2996.61 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-26 15:15:00 | 3037.75 | 3005.47 | 2996.81 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-09-28 10:15:00 | 2994.25 | 3005.90 | 2997.42 | SL hit qty=1.00 sl=2994.25 alert=retest2 |
| Cross detected — sustain check pending | 2023-09-29 10:15:00 | 3015.00 | 3004.81 | 2997.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-29 11:15:00 | 3023.85 | 3005.00 | 2997.29 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-10-03 09:15:00 | 2994.25 | 3006.48 | 2998.23 | SL hit qty=1.00 sl=2994.25 alert=retest2 |
| Cross detected — sustain check pending | 2023-10-03 10:15:00 | 3023.00 | 3006.65 | 2998.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-03 11:15:00 | 3019.35 | 3006.77 | 2998.46 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 09:15:00 | 3009.85 | 3007.20 | 2998.88 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2023-10-04 09:15:00 | 2994.25 | 3007.20 | 2998.88 | SL hit qty=1.00 sl=2994.25 alert=retest2 |
| Cross detected — sustain check pending | 2023-10-05 11:15:00 | 3016.80 | 3006.57 | 2998.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-10-05 12:15:00 | 3009.00 | 3006.59 | 2998.97 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-10-05 14:15:00 | 3017.55 | 3006.72 | 2999.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-05 15:15:00 | 3016.65 | 3006.82 | 2999.20 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-10-09 09:15:00 | 2990.50 | 3008.92 | 3000.57 | SL hit qty=1.00 sl=2990.50 alert=retest2 |
| Cross detected — sustain check pending | 2023-10-11 11:15:00 | 3018.00 | 3004.59 | 2998.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-11 12:15:00 | 3087.20 | 3005.42 | 2999.38 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2023-11-23 10:15:00 | 3550.28 | 3186.86 | 3122.89 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Target hit — 30% from entry | 2023-12-26 09:15:00 | 4013.36 | 3669.33 | 3474.18 | Target hit (30%) qty=0.50 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 09:15:00 | 5216.70 | 5528.42 | 5529.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 14:15:00 | 5152.35 | 5513.17 | 5521.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 09:15:00 | 4225.90 | 4190.34 | 4409.74 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 09:15:00 | 4374.20 | 4200.69 | 4407.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 4374.20 | 4200.69 | 4407.45 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-02-03 10:15:00 | 4289.05 | 4201.56 | 4406.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 11:15:00 | 4273.30 | 4202.28 | 4406.19 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-03-03 11:15:00 | 3632.30 | 3992.33 | 4184.43 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-15 12:15:00 | 4273.30 | 3851.68 | 3869.73 | SL hit qty=0.50 sl=4273.30 alert=retest2 |
| CROSSOVER_SKIP | 2025-05-16 13:15:00 | 4314.90 | 3887.26 | 3887.16 | HTF filter: close below htf_sma |
| Cross detected — sustain check pending | 2025-05-20 09:15:00 | 4327.80 | 3934.47 | 3911.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-20 10:15:00 | 4316.10 | 3938.27 | 3913.38 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-27 09:15:00 | 4302.80 | 4042.30 | 3973.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 10:15:00 | 4327.00 | 4045.13 | 3975.17 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-27 12:15:00 | 4333.10 | 4050.93 | 3978.78 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 13:15:00 | 4332.80 | 4053.74 | 3980.55 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 4336.70 | 4056.55 | 3982.32 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-30 14:15:00 | 4304.40 | 4111.84 | 4018.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 15:15:00 | 4309.30 | 4113.80 | 4020.25 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-09 12:15:00 | 4342.50 | 4153.61 | 4058.45 | SL hit qty=1.00 sl=4342.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-11 09:15:00 | 4428.40 | 4177.06 | 4075.57 | SL hit qty=1.00 sl=4428.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-11 09:15:00 | 4428.40 | 4177.06 | 4075.57 | SL hit qty=1.00 sl=4428.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-11 09:15:00 | 4428.40 | 4177.06 | 4075.57 | SL hit qty=1.00 sl=4428.40 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-13 09:15:00 | 4309.50 | 4205.70 | 4097.30 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 4300.40 | 4206.64 | 4098.31 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-13 14:15:00 | 4342.50 | 4210.99 | 4102.65 | SL hit qty=1.00 sl=4342.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-20 09:15:00 | 4282.40 | 4252.97 | 4140.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 10:15:00 | 4290.00 | 4253.33 | 4141.06 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-20 11:15:00 | 4342.50 | 4254.19 | 4142.05 | SL hit qty=1.00 sl=4342.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-20 13:15:00 | 4306.90 | 4255.38 | 4143.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-20 14:15:00 | 4332.40 | 4256.15 | 4144.71 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-23 09:15:00 | 4274.00 | 4257.13 | 4146.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 10:15:00 | 4269.20 | 4257.25 | 4146.93 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 4274.70 | 4257.43 | 4147.56 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-23 14:15:00 | 4257.00 | 4257.63 | 4149.30 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 15:15:00 | 4247.20 | 4257.53 | 4149.79 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 4282.10 | 4257.94 | 4150.53 | SL hit qty=1.00 sl=4282.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-25 09:15:00 | 4342.50 | 4261.31 | 4155.94 | SL hit qty=1.00 sl=4342.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-30 09:15:00 | 4232.80 | 4268.21 | 4170.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-30 10:15:00 | 4271.70 | 4268.25 | 4170.57 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-30 12:15:00 | 4248.50 | 4268.01 | 4171.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 13:15:00 | 4242.50 | 4267.76 | 4171.78 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-02 09:15:00 | 4282.10 | 4263.61 | 4174.33 | SL hit qty=1.00 sl=4282.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-02 13:15:00 | 4254.20 | 4263.20 | 4175.89 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 14:15:00 | 4245.50 | 4263.02 | 4176.24 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-03 09:15:00 | 4282.10 | 4263.01 | 4177.10 | SL hit qty=1.00 sl=4282.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-08 10:15:00 | 4248.00 | 4274.09 | 4191.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-08 11:15:00 | 4297.00 | 4274.32 | 4192.45 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-11 12:15:00 | 4232.30 | 4281.74 | 4204.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 13:15:00 | 4231.20 | 4281.24 | 4205.05 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 4231.50 | 4280.74 | 4205.19 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-07-11 15:15:00 | 4218.70 | 4280.12 | 4205.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 09:15:00 | 4219.20 | 4279.52 | 4205.32 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 3960m) |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 4241.40 | 4279.52 | 4205.32 | SL hit qty=1.00 sl=4241.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 4282.10 | 4277.66 | 4206.92 | SL hit qty=1.00 sl=4282.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-25 09:15:00 | 4201.00 | 4320.01 | 4249.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 10:15:00 | 4217.10 | 4318.99 | 4249.09 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-25 15:15:00 | 4221.40 | 4314.44 | 4248.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-28 09:15:00 | 4277.70 | 4314.07 | 4248.66 | ENTRY2 sustain failed after 3960m |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 4241.40 | 4314.07 | 4248.66 | SL hit qty=1.00 sl=4241.40 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-30 11:15:00 | 4227.50 | 4311.68 | 4252.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-30 12:15:00 | 4230.10 | 4310.86 | 4252.32 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-31 09:15:00 | 4218.00 | 4307.98 | 4252.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-31 10:15:00 | 4246.50 | 4307.37 | 4252.00 | ENTRY2 sustain failed after 60m |

### Cycle 4 — SELL (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 12:15:00 | 5379.00 | 5703.51 | 5705.06 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 11:15:00 | 5796.50 | 5701.05 | 5700.98 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2026-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 14:15:00 | 5593.50 | 5700.19 | 5700.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 15:15:00 | 5569.50 | 5698.89 | 5700.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 5653.50 | 5621.66 | 5656.61 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 5653.50 | 5621.66 | 5656.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 5653.50 | 5621.66 | 5656.61 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-03-04 09:15:00 | 5476.00 | 5639.46 | 5661.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 10:15:00 | 5468.50 | 5637.76 | 5660.99 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-04 14:15:00 | 5499.00 | 5631.55 | 5657.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 15:15:00 | 5500.00 | 5630.24 | 5656.60 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-09 09:15:00 | 5428.00 | 5615.50 | 5647.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 10:15:00 | 5395.00 | 5613.31 | 5645.81 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 5667.50 | 5605.95 | 5640.93 | SL hit qty=1.00 sl=5667.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 5667.50 | 5605.95 | 5640.93 | SL hit qty=1.00 sl=5667.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 5667.50 | 5605.95 | 5640.93 | SL hit qty=1.00 sl=5667.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-12 09:15:00 | 5419.50 | 5606.58 | 5639.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 10:15:00 | 5446.00 | 5604.98 | 5638.16 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 12:15:00 | 5460.00 | 5315.79 | 5435.78 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 5343.50 | 5320.34 | 5435.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 5269.50 | 5319.83 | 5434.88 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-09-22 10:15:00 | 3032.80 | 2023-09-25 09:15:00 | 2994.25 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2023-09-26 15:15:00 | 3037.75 | 2023-09-28 10:15:00 | 2994.25 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2023-09-29 11:15:00 | 3023.85 | 2023-10-03 09:15:00 | 2994.25 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2023-10-03 11:15:00 | 3019.35 | 2023-10-04 09:15:00 | 2994.25 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2023-10-05 15:15:00 | 3016.65 | 2023-10-09 09:15:00 | 2990.50 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2023-10-11 12:15:00 | 3087.20 | 2023-11-23 10:15:00 | 3550.28 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2023-10-11 12:15:00 | 3087.20 | 2023-12-26 09:15:00 | 4013.36 | TARGET_HIT | 0.50 | 30.00% |
| SELL | retest2 | 2025-02-03 11:15:00 | 4273.30 | 2025-03-03 11:15:00 | 3632.30 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-02-03 11:15:00 | 4273.30 | 2025-05-15 12:15:00 | 4273.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest2 | 2025-05-20 10:15:00 | 4316.10 | 2025-06-09 12:15:00 | 4342.50 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-05-27 10:15:00 | 4327.00 | 2025-06-11 09:15:00 | 4428.40 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-05-27 13:15:00 | 4332.80 | 2025-06-11 09:15:00 | 4428.40 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-05-30 15:15:00 | 4309.30 | 2025-06-11 09:15:00 | 4428.40 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2025-06-13 10:15:00 | 4300.40 | 2025-06-13 14:15:00 | 4342.50 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-06-20 10:15:00 | 4290.00 | 2025-06-20 11:15:00 | 4342.50 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-06-23 10:15:00 | 4269.20 | 2025-06-24 09:15:00 | 4282.10 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-06-23 15:15:00 | 4247.20 | 2025-06-25 09:15:00 | 4342.50 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-06-30 13:15:00 | 4242.50 | 2025-07-02 09:15:00 | 4282.10 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-07-02 14:15:00 | 4245.50 | 2025-07-03 09:15:00 | 4282.10 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-07-11 13:15:00 | 4231.20 | 2025-07-14 09:15:00 | 4241.40 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-07-14 09:15:00 | 4219.20 | 2025-07-15 09:15:00 | 4282.10 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-07-25 10:15:00 | 4217.10 | 2025-07-28 09:15:00 | 4241.40 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-03-04 10:15:00 | 5468.50 | 2026-03-10 10:15:00 | 5667.50 | STOP_HIT | 1.00 | -3.64% |
| SELL | retest2 | 2026-03-04 15:15:00 | 5500.00 | 2026-03-10 10:15:00 | 5667.50 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2026-03-09 10:15:00 | 5395.00 | 2026-03-10 10:15:00 | 5667.50 | STOP_HIT | 1.00 | -5.05% |
