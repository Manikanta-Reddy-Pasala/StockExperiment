# L&T Technology Services Ltd. (LTTS)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 3801.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 38 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 44 |
| PARTIAL | 15 |
| TARGET_HIT | 0 |
| STOP_HIT | 49 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 63 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 29 / 34
- **Target hits / Stop hits / Partials:** 0 / 48 / 15
- **Avg / median % per leg:** 0.19% / -1.01%
- **Sum % (uncompounded):** 11.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.27% | -8.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.27% | -8.9% |
| SELL (all) | 56 | 29 | 51.8% | 0 | 41 | 15 | 0.37% | 20.6% |
| SELL @ 2nd Alert (retest1) | 8 | 7 | 87.5% | 0 | 4 | 4 | 2.34% | 18.7% |
| SELL @ 3rd Alert (retest2) | 48 | 22 | 45.8% | 0 | 37 | 11 | 0.04% | 1.9% |
| retest1 (combined) | 8 | 7 | 87.5% | 0 | 4 | 4 | 2.34% | 18.7% |
| retest2 (combined) | 55 | 22 | 40.0% | 0 | 44 | 11 | -0.13% | -7.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 11:15:00 | 4432.90 | 4211.01 | 4210.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 13:15:00 | 4447.00 | 4215.46 | 4212.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 10:15:00 | 4487.80 | 4488.39 | 4393.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-29 11:00:00 | 4487.80 | 4488.39 | 4393.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 13:15:00 | 4361.00 | 4484.31 | 4402.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 14:00:00 | 4361.00 | 4484.31 | 4402.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 4375.50 | 4483.22 | 4402.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 15:15:00 | 4390.00 | 4483.22 | 4402.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 4345.80 | 4475.82 | 4401.81 | SL hit (close<static) qty=1.00 sl=4351.30 alert=retest2 |

### Cycle 2 — SELL (started 2026-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 15:15:00 | 3852.00 | 4354.86 | 4356.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 3823.70 | 4349.57 | 4354.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 15:15:00 | 3470.00 | 3457.53 | 3703.43 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 09:15:00 | 3270.00 | 3457.53 | 3703.43 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-18 11:45:00 | 3368.30 | 3437.51 | 3672.90 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-18 14:30:00 | 3366.30 | 3435.49 | 3668.37 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 09:15:00 | 3355.00 | 3434.88 | 3666.91 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 3199.89 | 3414.13 | 3639.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 3197.99 | 3414.13 | 3639.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 3187.25 | 3414.13 | 3639.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 12:15:00 | 3106.50 | 3405.69 | 3631.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 3349.70 | 3335.23 | 3548.35 | SL hit (close>ema200) qty=0.50 sl=3335.23 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-15 15:00:00 | 4495.00 | 2025-05-19 09:15:00 | 4542.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-05-16 11:15:00 | 4478.50 | 2025-05-19 09:15:00 | 4542.00 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-05-16 13:15:00 | 4491.00 | 2025-05-19 09:15:00 | 4542.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-05-20 12:30:00 | 4491.10 | 2025-05-23 09:15:00 | 4573.30 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-05-21 13:15:00 | 4440.00 | 2025-05-23 09:15:00 | 4573.30 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-05-21 13:45:00 | 4441.10 | 2025-05-23 09:15:00 | 4573.30 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2025-05-22 09:15:00 | 4435.10 | 2025-05-23 09:15:00 | 4573.30 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2025-05-27 10:15:00 | 4440.20 | 2025-06-10 14:15:00 | 4494.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-05-29 12:30:00 | 4415.10 | 2025-06-11 12:15:00 | 4545.90 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2025-05-29 14:00:00 | 4416.10 | 2025-06-11 12:15:00 | 4545.90 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2025-06-13 09:30:00 | 4420.80 | 2025-06-16 13:15:00 | 4534.00 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-06-13 13:15:00 | 4410.00 | 2025-06-16 13:15:00 | 4534.00 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-06-18 13:45:00 | 4421.40 | 2025-07-25 15:15:00 | 4200.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-19 09:45:00 | 4414.50 | 2025-07-25 15:15:00 | 4199.47 | PARTIAL | 0.50 | 4.87% |
| SELL | retest2 | 2025-07-17 12:30:00 | 4412.90 | 2025-07-29 09:15:00 | 4193.77 | PARTIAL | 0.50 | 4.97% |
| SELL | retest2 | 2025-07-17 14:00:00 | 4420.50 | 2025-07-29 09:15:00 | 4192.25 | PARTIAL | 0.50 | 5.16% |
| SELL | retest2 | 2025-06-18 13:45:00 | 4421.40 | 2025-08-21 09:15:00 | 4264.80 | STOP_HIT | 0.50 | 3.54% |
| SELL | retest2 | 2025-06-19 09:45:00 | 4414.50 | 2025-08-21 09:15:00 | 4264.80 | STOP_HIT | 0.50 | 3.39% |
| SELL | retest2 | 2025-07-17 12:30:00 | 4412.90 | 2025-08-21 09:15:00 | 4264.80 | STOP_HIT | 0.50 | 3.36% |
| SELL | retest2 | 2025-07-17 14:00:00 | 4420.50 | 2025-08-21 09:15:00 | 4264.80 | STOP_HIT | 0.50 | 3.52% |
| SELL | retest2 | 2025-07-23 09:15:00 | 4307.00 | 2025-08-25 09:15:00 | 4364.00 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-07-23 15:15:00 | 4304.00 | 2025-09-05 12:15:00 | 4091.65 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2025-07-24 11:15:00 | 4299.00 | 2025-09-08 09:15:00 | 4088.80 | PARTIAL | 0.50 | 4.89% |
| SELL | retest2 | 2025-07-31 13:45:00 | 4300.00 | 2025-09-08 09:15:00 | 4085.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-22 10:15:00 | 4242.30 | 2025-09-09 15:15:00 | 4084.05 | PARTIAL | 0.50 | 3.73% |
| SELL | retest2 | 2025-07-23 15:15:00 | 4304.00 | 2025-09-10 09:15:00 | 4234.60 | STOP_HIT | 0.50 | 1.61% |
| SELL | retest2 | 2025-07-24 11:15:00 | 4299.00 | 2025-09-10 09:15:00 | 4234.60 | STOP_HIT | 0.50 | 1.50% |
| SELL | retest2 | 2025-07-31 13:45:00 | 4300.00 | 2025-09-10 09:15:00 | 4234.60 | STOP_HIT | 0.50 | 1.52% |
| SELL | retest2 | 2025-08-22 10:15:00 | 4242.30 | 2025-09-10 09:15:00 | 4234.60 | STOP_HIT | 0.50 | 0.18% |
| SELL | retest2 | 2025-08-28 15:00:00 | 4215.30 | 2025-09-18 09:15:00 | 4467.10 | STOP_HIT | 1.00 | -5.97% |
| SELL | retest2 | 2025-08-29 11:30:00 | 4238.20 | 2025-09-18 09:15:00 | 4467.10 | STOP_HIT | 1.00 | -5.40% |
| SELL | retest2 | 2025-09-01 12:30:00 | 4242.60 | 2025-09-18 09:15:00 | 4467.10 | STOP_HIT | 1.00 | -5.29% |
| SELL | retest2 | 2025-09-22 09:45:00 | 4250.50 | 2025-09-26 13:15:00 | 4037.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 14:15:00 | 4248.50 | 2025-09-26 13:15:00 | 4036.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 09:45:00 | 4250.50 | 2025-10-03 09:15:00 | 4208.00 | STOP_HIT | 0.50 | 1.00% |
| SELL | retest2 | 2025-09-22 14:15:00 | 4248.50 | 2025-10-03 09:15:00 | 4208.00 | STOP_HIT | 0.50 | 0.95% |
| SELL | retest2 | 2025-10-06 09:15:00 | 4248.50 | 2025-10-06 10:15:00 | 4324.90 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-10-08 12:00:00 | 4254.50 | 2025-11-07 09:15:00 | 4041.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-08 12:00:00 | 4254.50 | 2025-11-12 10:15:00 | 4190.00 | STOP_HIT | 0.50 | 1.52% |
| SELL | retest2 | 2025-10-23 11:15:00 | 4197.40 | 2025-11-19 10:15:00 | 4272.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-10-23 15:15:00 | 4196.00 | 2025-11-19 10:15:00 | 4272.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-10-24 10:00:00 | 4197.60 | 2025-11-19 10:15:00 | 4272.00 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-10-27 15:00:00 | 4193.90 | 2025-11-19 10:15:00 | 4272.00 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-11-12 13:15:00 | 4181.20 | 2025-11-19 10:15:00 | 4272.00 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2026-01-01 15:15:00 | 4390.00 | 2026-01-05 09:15:00 | 4345.80 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2026-01-05 13:15:00 | 4396.00 | 2026-01-06 10:15:00 | 4339.40 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2026-01-06 09:45:00 | 4395.00 | 2026-01-06 10:15:00 | 4339.40 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2026-01-07 14:30:00 | 4394.10 | 2026-01-08 10:15:00 | 4357.30 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2026-01-08 09:30:00 | 4408.00 | 2026-01-09 13:15:00 | 4363.80 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2026-01-08 14:00:00 | 4426.60 | 2026-01-09 13:15:00 | 4363.80 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2026-01-09 09:30:00 | 4405.00 | 2026-01-12 09:15:00 | 4313.10 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest1 | 2026-03-16 09:15:00 | 3270.00 | 2026-03-23 09:15:00 | 3199.89 | PARTIAL | 0.50 | 2.14% |
| SELL | retest1 | 2026-03-18 11:45:00 | 3368.30 | 2026-03-23 09:15:00 | 3197.99 | PARTIAL | 0.50 | 5.06% |
| SELL | retest1 | 2026-03-18 14:30:00 | 3366.30 | 2026-03-23 09:15:00 | 3187.25 | PARTIAL | 0.50 | 5.32% |
| SELL | retest1 | 2026-03-19 09:15:00 | 3355.00 | 2026-03-23 12:15:00 | 3106.50 | PARTIAL | 0.50 | 7.41% |
| SELL | retest1 | 2026-03-16 09:15:00 | 3270.00 | 2026-04-02 13:15:00 | 3349.70 | STOP_HIT | 0.50 | -2.44% |
| SELL | retest1 | 2026-03-18 11:45:00 | 3368.30 | 2026-04-02 13:15:00 | 3349.70 | STOP_HIT | 0.50 | 0.55% |
| SELL | retest1 | 2026-03-18 14:30:00 | 3366.30 | 2026-04-02 13:15:00 | 3349.70 | STOP_HIT | 0.50 | 0.49% |
| SELL | retest1 | 2026-03-19 09:15:00 | 3355.00 | 2026-04-02 13:15:00 | 3349.70 | STOP_HIT | 0.50 | 0.16% |
| SELL | retest2 | 2026-04-16 10:15:00 | 3447.60 | 2026-04-21 09:15:00 | 3607.00 | STOP_HIT | 1.00 | -4.62% |
| SELL | retest2 | 2026-04-20 09:30:00 | 3448.70 | 2026-04-21 09:15:00 | 3607.00 | STOP_HIT | 1.00 | -4.59% |
| SELL | retest2 | 2026-04-23 10:30:00 | 3451.00 | 2026-04-27 14:15:00 | 3592.10 | STOP_HIT | 1.00 | -4.09% |
| SELL | retest2 | 2026-04-23 12:15:00 | 3447.00 | 2026-04-27 14:15:00 | 3592.10 | STOP_HIT | 1.00 | -4.21% |
