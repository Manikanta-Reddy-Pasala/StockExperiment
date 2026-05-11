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
| ALERT2_SKIP | 2 |
| ALERT3 | 27 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 32 |
| PARTIAL | 11 |
| TARGET_HIT | 0 |
| STOP_HIT | 33 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 43 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 21
- **Target hits / Stop hits / Partials:** 0 / 32 / 11
- **Avg / median % per leg:** 0.50% / 0.18%
- **Sum % (uncompounded):** 21.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 43 | 22 | 51.2% | 0 | 32 | 11 | 0.50% | 21.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 43 | 22 | 51.2% | 0 | 32 | 11 | 0.50% | 21.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 43 | 22 | 51.2% | 0 | 32 | 11 | 0.50% | 21.6% |

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
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 13:15:00 | 4361.00 | 4484.31 | 4402.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 13:15:00 | 4361.00 | 4484.31 | 4402.16 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2026-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 15:15:00 | 3852.00 | 4354.86 | 4356.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 3823.70 | 4349.57 | 4354.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 15:15:00 | 3470.00 | 3457.53 | 3703.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 09:15:00 | 3456.00 | 3357.36 | 3511.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 3456.00 | 3357.36 | 3511.55 | EMA400 retest candle locked (from downside) |


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
