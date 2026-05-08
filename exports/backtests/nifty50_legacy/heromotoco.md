# HEROMOTOCO (HEROMOTOCO)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:15:00 (4996 bars)
- **Last close:** 5322.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 3 |
| ALERT3 | 6 |
| PENDING | 23 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 4 |
| ENTRY2 | 15 |
| PARTIAL | 9 |
| TARGET_HIT | 8 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 18 / 7
- **Target hits / Stop hits / Partials:** 8 / 8 / 9
- **Avg / median % per leg:** 14.63% / 15.00%
- **Sum % (uncompounded):** 365.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 16 | 80.0% | 8 | 4 | 8 | 17.56% | 351.2% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.20% | -8.8% |
| BUY @ 3rd Alert (retest2) | 16 | 16 | 100.0% | 8 | 0 | 8 | 22.50% | 360.0% |
| SELL (all) | 5 | 2 | 40.0% | 0 | 4 | 1 | 2.92% | 14.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 2 | 40.0% | 0 | 4 | 1 | 2.92% | 14.6% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.20% | -8.8% |
| retest2 (combined) | 21 | 18 | 85.7% | 8 | 4 | 9 | 17.84% | 374.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-20 10:15:00 | 3117.75 | 3001.73 | 3001.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-16 09:15:00 | 3143.40 | 3021.74 | 3012.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-30 09:15:00 | 3082.90 | 3085.82 | 3052.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 09:15:00 | 3062.65 | 3088.21 | 3057.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 09:15:00 | 3062.65 | 3088.21 | 3057.51 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2023-11-06 09:15:00 | 3147.00 | 3087.56 | 3059.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 10:15:00 | 3164.85 | 3088.33 | 3059.74 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2023-11-08 09:15:00 | 3143.15 | 3095.67 | 3065.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-11-08 10:15:00 | 3140.00 | 3096.11 | 3065.71 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-11-09 09:15:00 | 3178.30 | 3098.83 | 3067.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-09 10:15:00 | 3177.35 | 3099.61 | 3068.54 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2023-11-10 11:15:00 | 3148.00 | 3104.31 | 3072.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-11-10 12:15:00 | 3119.25 | 3104.46 | 3072.40 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-11-15 09:15:00 | 3165.05 | 3106.37 | 3075.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-15 10:15:00 | 3157.60 | 3106.88 | 3075.50 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-29 09:15:00 | 3639.58 | 3258.53 | 3168.14 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-29 09:15:00 | 3653.95 | 3258.53 | 3168.14 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-29 09:15:00 | 3631.24 | 3258.53 | 3168.14 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Target hit | 2023-12-28 09:15:00 | 4114.31 | 3720.28 | 3514.54 | Target hit (30%) qty=0.50 alert=retest2 |
| Target hit | 2023-12-28 09:15:00 | 4130.56 | 3720.28 | 3514.54 | Target hit (30%) qty=0.50 alert=retest2 |
| Target hit | 2023-12-28 09:15:00 | 4104.88 | 3720.28 | 3514.54 | Target hit (30%) qty=0.50 alert=retest2 |

### Cycle 2 — SELL (started 2024-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 09:15:00 | 5216.70 | 5528.42 | 5529.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 14:15:00 | 5152.35 | 5513.17 | 5521.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 09:15:00 | 4225.90 | 4190.34 | 4409.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 09:15:00 | 4374.20 | 4200.69 | 4407.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 4374.20 | 4200.69 | 4407.45 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-02-03 10:15:00 | 4289.05 | 4201.56 | 4406.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 11:15:00 | 4273.30 | 4202.28 | 4406.19 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 11:15:00 | 3632.30 | 3992.33 | 4184.43 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-27 10:15:00 | 3746.95 | 3724.32 | 3925.39 | SL hit (close>ema200) qty=0.50 sl=3724.32 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 13:15:00 | 4314.90 | 3887.26 | 3887.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 14:15:00 | 4352.10 | 3891.88 | 3889.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 14:15:00 | 4257.00 | 4257.63 | 4149.30 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-06-24 09:15:00 | 4299.00 | 4257.94 | 4150.53 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 10:15:00 | 4311.20 | 4258.47 | 4151.33 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-26 14:15:00 | 4283.90 | 4264.13 | 4163.55 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 15:15:00 | 4281.00 | 4264.30 | 4164.13 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-30 10:15:00 | 4271.70 | 4268.25 | 4170.57 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-30 11:15:00 | 4264.00 | 4268.21 | 4171.04 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-03 09:15:00 | 4287.00 | 4263.01 | 4177.10 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 10:15:00 | 4284.70 | 4263.23 | 4177.64 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-08 11:15:00 | 4297.00 | 4274.32 | 4192.45 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-08 12:15:00 | 4305.40 | 4274.63 | 4193.02 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 4218.70 | 4280.12 | 4205.25 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-07-14 12:15:00 | 4234.00 | 4278.05 | 4205.69 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 13:15:00 | 4238.80 | 4277.66 | 4205.86 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 4201.00 | 4320.01 | 4249.25 | SL hit (close<ema400) qty=1.00 sl=4249.25 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 4201.00 | 4320.01 | 4249.25 | SL hit (close<ema400) qty=1.00 sl=4249.25 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 4201.00 | 4320.01 | 4249.25 | SL hit (close<ema400) qty=1.00 sl=4249.25 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 4201.00 | 4320.01 | 4249.25 | SL hit (close<ema400) qty=1.00 sl=4249.25 alert=retest1 |
| Cross detected — sustain check pending | 2025-07-25 14:15:00 | 4235.40 | 4315.37 | 4248.65 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-25 15:15:00 | 4221.40 | 4314.44 | 4248.51 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-28 09:15:00 | 4277.70 | 4314.07 | 4248.66 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 10:15:00 | 4279.40 | 4313.73 | 4248.81 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-30 13:15:00 | 4241.70 | 4310.18 | 4252.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 14:15:00 | 4249.40 | 4309.57 | 4252.26 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-31 10:15:00 | 4246.50 | 4307.37 | 4252.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:15:00 | 4264.80 | 4306.95 | 4252.07 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 4251.90 | 4305.53 | 4252.44 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-08-01 09:15:00 | 4259.10 | 4305.06 | 4252.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 10:15:00 | 4290.20 | 4304.92 | 4252.66 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 09:15:00 | 4874.62 | 4457.77 | 4354.08 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 09:15:00 | 4921.31 | 4457.77 | 4354.08 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 09:15:00 | 4886.81 | 4457.77 | 4354.08 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 09:15:00 | 4904.52 | 4457.77 | 4354.08 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 09:15:00 | 4933.73 | 4457.77 | 4354.08 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-09-08 10:15:00 | 5510.44 | 4927.43 | 4679.72 | Target hit (30%) qty=0.50 alert=retest2 |
| Target hit | 2025-09-08 10:15:00 | 5524.22 | 4927.43 | 4679.72 | Target hit (30%) qty=0.50 alert=retest2 |
| Target hit | 2025-10-03 11:15:00 | 5544.24 | 5250.59 | 5006.09 | Target hit (30%) qty=0.50 alert=retest2 |
| Target hit | 2025-10-03 12:15:00 | 5563.22 | 5253.63 | 5008.83 | Target hit (30%) qty=0.50 alert=retest2 |
| Target hit | 2025-10-03 12:15:00 | 5577.26 | 5253.63 | 5008.83 | Target hit (30%) qty=0.50 alert=retest2 |

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
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 5653.50 | 5621.66 | 5656.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 5653.50 | 5621.66 | 5656.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 5653.50 | 5621.66 | 5656.61 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-03-04 09:15:00 | 5476.00 | 5639.46 | 5661.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 10:15:00 | 5468.50 | 5637.76 | 5660.99 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-04 14:15:00 | 5499.00 | 5631.55 | 5657.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 15:15:00 | 5500.00 | 5630.24 | 5656.60 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-09 09:15:00 | 5428.00 | 5615.50 | 5647.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 10:15:00 | 5395.00 | 5613.31 | 5645.81 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 5685.00 | 5605.95 | 5640.93 | SL hit (close>static) qty=1.00 sl=5667.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 5685.00 | 5605.95 | 5640.93 | SL hit (close>static) qty=1.00 sl=5667.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 5685.00 | 5605.95 | 5640.93 | SL hit (close>static) qty=1.00 sl=5667.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-12 09:15:00 | 5419.50 | 5606.58 | 5639.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 10:15:00 | 5446.00 | 5604.98 | 5638.16 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 12:15:00 | 5460.00 | 5315.79 | 5435.78 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 5343.50 | 5320.34 | 5435.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 5269.50 | 5319.83 | 5434.88 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-05-07 13:15:00 | 5379.50 | 5200.97 | 5312.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 14:15:00 | 5335.50 | 5202.30 | 5312.23 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-11-06 10:15:00 | 3164.85 | 2023-11-29 09:15:00 | 3639.58 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2023-11-09 10:15:00 | 3177.35 | 2023-11-29 09:15:00 | 3653.95 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2023-11-15 10:15:00 | 3157.60 | 2023-11-29 09:15:00 | 3631.24 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2023-11-06 10:15:00 | 3164.85 | 2023-12-28 09:15:00 | 4114.31 | TARGET_HIT | 0.50 | 30.00% |
| BUY | retest2 | 2023-11-09 10:15:00 | 3177.35 | 2023-12-28 09:15:00 | 4130.56 | TARGET_HIT | 0.50 | 30.00% |
| BUY | retest2 | 2023-11-15 10:15:00 | 3157.60 | 2023-12-28 09:15:00 | 4104.88 | TARGET_HIT | 0.50 | 30.00% |
| SELL | retest2 | 2025-02-03 11:15:00 | 4273.30 | 2025-03-03 11:15:00 | 3632.30 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-02-03 11:15:00 | 4273.30 | 2025-03-27 10:15:00 | 3746.95 | STOP_HIT | 0.50 | 12.32% |
| BUY | retest1 | 2025-06-24 10:15:00 | 4311.20 | 2025-07-25 09:15:00 | 4201.00 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest1 | 2025-06-26 15:15:00 | 4281.00 | 2025-07-25 09:15:00 | 4201.00 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest1 | 2025-07-03 10:15:00 | 4284.70 | 2025-07-25 09:15:00 | 4201.00 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest1 | 2025-07-08 12:15:00 | 4305.40 | 2025-07-25 09:15:00 | 4201.00 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2025-07-14 13:15:00 | 4238.80 | 2025-08-18 09:15:00 | 4874.62 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-07-28 10:15:00 | 4279.40 | 2025-08-18 09:15:00 | 4921.31 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-07-30 14:15:00 | 4249.40 | 2025-08-18 09:15:00 | 4886.81 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-07-31 11:15:00 | 4264.80 | 2025-08-18 09:15:00 | 4904.52 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-08-01 10:15:00 | 4290.20 | 2025-08-18 09:15:00 | 4933.73 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-07-14 13:15:00 | 4238.80 | 2025-09-08 10:15:00 | 5510.44 | TARGET_HIT | 0.50 | 30.00% |
| BUY | retest2 | 2025-07-28 10:15:00 | 4279.40 | 2025-09-08 10:15:00 | 5524.22 | TARGET_HIT | 0.50 | 29.09% |
| BUY | retest2 | 2025-07-30 14:15:00 | 4249.40 | 2025-10-03 11:15:00 | 5544.24 | TARGET_HIT | 0.50 | 30.47% |
| BUY | retest2 | 2025-07-31 11:15:00 | 4264.80 | 2025-10-03 12:15:00 | 5563.22 | TARGET_HIT | 0.50 | 30.45% |
| BUY | retest2 | 2025-08-01 10:15:00 | 4290.20 | 2025-10-03 12:15:00 | 5577.26 | TARGET_HIT | 0.50 | 30.00% |
| SELL | retest2 | 2026-03-04 10:15:00 | 5468.50 | 2026-03-10 10:15:00 | 5685.00 | STOP_HIT | 1.00 | -3.96% |
| SELL | retest2 | 2026-03-04 15:15:00 | 5500.00 | 2026-03-10 10:15:00 | 5685.00 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2026-03-09 10:15:00 | 5395.00 | 2026-03-10 10:15:00 | 5685.00 | STOP_HIT | 1.00 | -5.38% |
