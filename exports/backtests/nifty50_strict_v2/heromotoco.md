# HEROMOTOCO (HEROMOTOCO)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:15:00 (4996 bars)
- **Last close:** 5322.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty booked @ 5% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
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
| PARTIAL | 0 |
| TARGET_HIT | 9 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 9 / 7
- **Target hits / Stop hits / Partials:** 9 / 7 / 0
- **Avg / median % per leg:** 4.28% / 10.00%
- **Sum % (uncompounded):** 68.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 8 | 66.7% | 8 | 4 | 0 | 5.93% | 71.2% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.20% | -8.8% |
| BUY @ 3rd Alert (retest2) | 8 | 8 | 100.0% | 8 | 0 | 0 | 10.00% | 80.0% |
| SELL (all) | 4 | 1 | 25.0% | 1 | 3 | 0 | -0.67% | -2.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 1 | 25.0% | 1 | 3 | 0 | -0.67% | -2.7% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.20% | -8.8% |
| retest2 (combined) | 12 | 9 | 75.0% | 9 | 3 | 0 | 6.44% | 77.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-20 09:15:00 | 3110.75 | 2997.93 | 2997.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-16 09:15:00 | 3143.40 | 3020.94 | 3011.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-30 09:15:00 | 3082.90 | 3085.39 | 3051.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 09:15:00 | 3062.65 | 3087.87 | 3056.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 09:15:00 | 3062.65 | 3087.87 | 3056.48 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2023-11-06 09:15:00 | 3147.00 | 3087.26 | 3058.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 10:15:00 | 3164.85 | 3088.03 | 3058.78 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2023-11-08 09:15:00 | 3143.15 | 3095.41 | 3064.45 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-11-08 10:15:00 | 3140.00 | 3095.85 | 3064.82 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-11-09 09:15:00 | 3178.30 | 3098.59 | 3067.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-09 10:15:00 | 3177.35 | 3099.37 | 3067.68 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2023-11-10 11:15:00 | 3148.00 | 3104.09 | 3071.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-11-10 12:15:00 | 3119.25 | 3104.24 | 3071.58 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-11-15 09:15:00 | 3165.05 | 3106.18 | 3074.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-15 10:15:00 | 3157.60 | 3106.69 | 3074.72 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Target hit | 2023-11-23 09:15:00 | 3481.34 | 3183.24 | 3122.02 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2023-11-23 09:15:00 | 3495.09 | 3183.24 | 3122.02 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2023-11-23 09:15:00 | 3473.36 | 3183.24 | 3122.02 | Target hit (10%) qty=1.00 alert=retest2 |

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
| Target hit | 2025-02-17 09:15:00 | 3845.97 | 4158.82 | 4325.85 | Target hit (10%) qty=1.00 alert=retest2 |

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
| Target hit | 2025-08-07 14:15:00 | 4662.68 | 4356.76 | 4287.91 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-08-08 09:15:00 | 4707.34 | 4362.61 | 4291.53 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-08-08 09:15:00 | 4674.34 | 4362.61 | 4291.53 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-08-08 09:15:00 | 4691.28 | 4362.61 | 4291.53 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-08-13 09:15:00 | 4719.22 | 4410.73 | 4323.88 | Target hit (10%) qty=1.00 alert=retest2 |

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
| BUY | retest2 | 2023-11-06 10:15:00 | 3164.85 | 2023-11-23 09:15:00 | 3481.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-09 10:15:00 | 3177.35 | 2023-11-23 09:15:00 | 3495.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-15 10:15:00 | 3157.60 | 2023-11-23 09:15:00 | 3473.36 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-03 11:15:00 | 4273.30 | 2025-02-17 09:15:00 | 3845.97 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2025-06-24 10:15:00 | 4311.20 | 2025-07-25 09:15:00 | 4201.00 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest1 | 2025-06-26 15:15:00 | 4281.00 | 2025-07-25 09:15:00 | 4201.00 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest1 | 2025-07-03 10:15:00 | 4284.70 | 2025-07-25 09:15:00 | 4201.00 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest1 | 2025-07-08 12:15:00 | 4305.40 | 2025-07-25 09:15:00 | 4201.00 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2025-07-14 13:15:00 | 4238.80 | 2025-08-07 14:15:00 | 4662.68 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-28 10:15:00 | 4279.40 | 2025-08-08 09:15:00 | 4707.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-30 14:15:00 | 4249.40 | 2025-08-08 09:15:00 | 4674.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-31 11:15:00 | 4264.80 | 2025-08-08 09:15:00 | 4691.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-01 10:15:00 | 4290.20 | 2025-08-13 09:15:00 | 4719.22 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-04 10:15:00 | 5468.50 | 2026-03-10 10:15:00 | 5685.00 | STOP_HIT | 1.00 | -3.96% |
| SELL | retest2 | 2026-03-04 15:15:00 | 5500.00 | 2026-03-10 10:15:00 | 5685.00 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2026-03-09 10:15:00 | 5395.00 | 2026-03-10 10:15:00 | 5685.00 | STOP_HIT | 1.00 | -5.38% |
