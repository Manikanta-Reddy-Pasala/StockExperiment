# HEROMOTOCO (HEROMOTOCO)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 5356.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 4 |
| PENDING | 12 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 6 |
| PARTIAL | 5 |
| TARGET_HIT | 5 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 5
- **Target hits / Stop hits / Partials:** 5 / 5 / 5
- **Avg / median % per leg:** 14.27% / 15.00%
- **Sum % (uncompounded):** 213.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 10 | 66.7% | 5 | 5 | 5 | 14.27% | 214.0% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.15% | -8.6% |
| BUY @ 3rd Alert (retest2) | 11 | 10 | 90.9% | 5 | 1 | 5 | 20.24% | 222.6% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.15% | -8.6% |
| retest2 (combined) | 11 | 10 | 90.9% | 5 | 1 | 5 | 20.24% | 222.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 13:15:00 | 5688.20 | 5368.70 | 5368.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 10:15:00 | 5704.75 | 5399.31 | 5383.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 10:15:00 | 5741.00 | 5772.57 | 5622.59 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 13:15:00 | 5645.80 | 5762.44 | 5629.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 5645.80 | 5762.44 | 5629.55 | EMA400 retest candle locked |

### Cycle 2 — BUY (started 2025-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 13:15:00 | 4314.90 | 3887.56 | 3887.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 14:15:00 | 4354.60 | 3892.20 | 3889.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 14:15:00 | 4256.90 | 4257.48 | 4149.20 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-06-24 09:15:00 | 4299.00 | 4257.79 | 4150.44 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 10:15:00 | 4311.30 | 4258.32 | 4151.24 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-26 14:15:00 | 4283.90 | 4263.98 | 4163.45 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 15:15:00 | 4281.00 | 4264.15 | 4164.03 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-30 10:15:00 | 4271.70 | 4268.08 | 4170.46 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-30 11:15:00 | 4264.60 | 4268.05 | 4170.93 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-03 09:15:00 | 4286.00 | 4262.88 | 4177.00 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 10:15:00 | 4284.70 | 4263.10 | 4177.54 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-08 11:15:00 | 4297.00 | 4274.21 | 4192.36 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-08 12:15:00 | 4305.40 | 4274.52 | 4192.93 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 4203.00 | 4279.86 | 4205.08 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-11 15:15:00 | 4203.00 | 4279.86 | 4205.08 | SL hit (close<ema400) qty=1.00 sl=4205.08 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-11 15:15:00 | 4203.00 | 4279.86 | 4205.08 | SL hit (close<ema400) qty=1.00 sl=4205.08 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-11 15:15:00 | 4203.00 | 4279.86 | 4205.08 | SL hit (close<ema400) qty=1.00 sl=4205.08 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-11 15:15:00 | 4203.00 | 4279.86 | 4205.08 | SL hit (close<ema400) qty=1.00 sl=4205.08 alert=retest1 |
| Cross detected — sustain check pending | 2025-07-14 12:15:00 | 4234.00 | 4277.80 | 4205.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 13:15:00 | 4238.70 | 4277.41 | 4205.69 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-25 14:15:00 | 4234.10 | 4315.20 | 4248.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-25 15:15:00 | 4221.40 | 4314.27 | 4248.36 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-28 09:15:00 | 4277.60 | 4313.90 | 4248.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 10:15:00 | 4279.40 | 4313.56 | 4248.66 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-30 13:15:00 | 4241.70 | 4310.05 | 4252.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 14:15:00 | 4249.60 | 4309.45 | 4252.14 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-31 10:15:00 | 4246.50 | 4307.33 | 4251.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:15:00 | 4264.80 | 4306.91 | 4251.99 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 4251.90 | 4305.49 | 4252.36 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-08-01 10:15:00 | 4290.20 | 4304.88 | 4252.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 11:15:00 | 4297.00 | 4304.80 | 4252.80 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-08-18 09:15:00 | 4874.50 | 4457.55 | 4353.91 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-08-18 09:15:00 | 4921.31 | 4457.55 | 4353.91 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-08-18 09:15:00 | 4887.04 | 4457.55 | 4353.91 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-08-18 09:15:00 | 4904.52 | 4457.55 | 4353.91 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-08-18 09:15:00 | 4941.55 | 4457.55 | 4353.91 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Target hit — 30% from entry | 2025-09-04 09:15:00 | 5510.31 | 4853.60 | 4624.93 | Target hit (30%) qty=0.50 alert=retest2 |
| Target hit — 30% from entry | 2025-09-08 10:15:00 | 5524.48 | 4927.75 | 4679.87 | Target hit (30%) qty=0.50 alert=retest2 |
| Target hit — 30% from entry | 2025-10-03 11:15:00 | 5544.24 | 5251.25 | 5006.50 | Target hit (30%) qty=0.50 alert=retest2 |
| Target hit — 30% from entry | 2025-10-03 12:15:00 | 5563.22 | 5254.30 | 5009.25 | Target hit (30%) qty=0.50 alert=retest2 |
| Target hit — 30% from entry | 2025-10-06 09:15:00 | 5586.10 | 5265.83 | 5019.92 | Target hit (30%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2026-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 15:15:00 | 5754.50 | 5699.06 | 5698.99 | EMA200 above EMA400 |

### Cycle 4 — BUY (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 11:15:00 | 5744.00 | 5699.29 | 5699.10 | EMA200 above EMA400 |

### Cycle 5 — BUY (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 09:15:00 | 5778.00 | 5699.61 | 5699.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 11:15:00 | 5813.00 | 5622.86 | 5655.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 10:15:00 | 5607.00 | 5642.68 | 5663.01 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 10:15:00 | 5607.00 | 5642.68 | 5663.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 10:15:00 | 5607.00 | 5642.68 | 5663.01 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-03-10 14:15:00 | 5704.00 | 5606.53 | 5639.39 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 15:15:00 | 5716.00 | 5607.62 | 5639.77 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-11 12:15:00 | 5578.50 | 5608.05 | 5639.36 | SL hit (close<static) qty=1.00 sl=5581.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-24 10:15:00 | 4311.30 | 2025-07-11 15:15:00 | 4203.00 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest1 | 2025-06-26 15:15:00 | 4281.00 | 2025-07-11 15:15:00 | 4203.00 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest1 | 2025-07-03 10:15:00 | 4284.70 | 2025-07-11 15:15:00 | 4203.00 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest1 | 2025-07-08 12:15:00 | 4305.40 | 2025-07-11 15:15:00 | 4203.00 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2025-07-14 13:15:00 | 4238.70 | 2025-08-18 09:15:00 | 4874.50 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-07-28 10:15:00 | 4279.40 | 2025-08-18 09:15:00 | 4921.31 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-07-30 14:15:00 | 4249.60 | 2025-08-18 09:15:00 | 4887.04 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-07-31 11:15:00 | 4264.80 | 2025-08-18 09:15:00 | 4904.52 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-08-01 11:15:00 | 4297.00 | 2025-08-18 09:15:00 | 4941.55 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-07-14 13:15:00 | 4238.70 | 2025-09-04 09:15:00 | 5510.31 | TARGET_HIT | 0.50 | 30.00% |
| BUY | retest2 | 2025-07-28 10:15:00 | 4279.40 | 2025-09-08 10:15:00 | 5524.48 | TARGET_HIT | 0.50 | 29.09% |
| BUY | retest2 | 2025-07-30 14:15:00 | 4249.60 | 2025-10-03 11:15:00 | 5544.24 | TARGET_HIT | 0.50 | 30.46% |
| BUY | retest2 | 2025-07-31 11:15:00 | 4264.80 | 2025-10-03 12:15:00 | 5563.22 | TARGET_HIT | 0.50 | 30.45% |
| BUY | retest2 | 2025-08-01 11:15:00 | 4297.00 | 2025-10-06 09:15:00 | 5586.10 | TARGET_HIT | 0.50 | 30.00% |
| BUY | retest2 | 2026-03-10 15:15:00 | 5716.00 | 2026-03-11 12:15:00 | 5578.50 | STOP_HIT | 1.00 | -2.41% |
