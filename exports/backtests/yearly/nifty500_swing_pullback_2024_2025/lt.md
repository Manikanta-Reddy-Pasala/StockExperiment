# Larsen & Toubro Ltd. (LT)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 3974.50
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
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 0 / 4 / 1
- **Avg / median % per leg:** -0.13% / -2.65%
- **Sum % (uncompounded):** -0.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 0 | 4 | 1 | -0.13% | -0.6% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 4 | 1 | -0.13% | -0.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 0 | 4 | 1 | -0.13% | -0.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 05:30:00 | 3702.70 | 3458.66 | 3614.47 | Stage2 pullback-breakout RSI=59 vol=1.7x ATR=67.41 |
| Stop hit — per-position SL triggered | 2024-09-06 05:30:00 | 3601.58 | 3474.27 | 3636.41 | SL hit (bars_held=8) |

### Cycle 2 — BUY (started 2024-09-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 05:30:00 | 3793.90 | 3491.29 | 3657.83 | Stage2 pullback-breakout RSI=64 vol=2.5x ATR=69.27 |
| Stop hit — per-position SL triggered | 2024-09-27 05:30:00 | 3689.99 | 3504.77 | 3699.70 | SL hit (bars_held=5) |

### Cycle 3 — BUY (started 2024-10-31 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 05:30:00 | 3622.30 | 3504.23 | 3502.23 | Stage2 pullback-breakout RSI=58 vol=3.1x ATR=89.96 |
| Stop hit — per-position SL triggered | 2024-11-14 05:30:00 | 3526.25 | 3513.51 | 3561.56 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-11-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 05:30:00 | 3753.00 | 3516.69 | 3572.16 | Stage2 pullback-breakout RSI=64 vol=2.2x ATR=86.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 05:30:00 | 3926.69 | 3541.19 | 3713.05 | T1 booked 50% @ 3926.69 |
| Stop hit — per-position SL triggered | 2024-12-12 05:30:00 | 3859.90 | 3551.78 | 3760.97 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-27 05:30:00 | 3702.70 | 2024-09-06 05:30:00 | 3601.58 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest1 | 2024-09-20 05:30:00 | 3793.90 | 2024-09-27 05:30:00 | 3689.99 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest1 | 2024-10-31 05:30:00 | 3622.30 | 2024-11-14 05:30:00 | 3526.25 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest1 | 2024-11-25 05:30:00 | 3753.00 | 2024-12-09 05:30:00 | 3926.69 | PARTIAL | 0.50 | 4.63% |
| BUY | retest1 | 2024-11-25 05:30:00 | 3753.00 | 2024-12-12 05:30:00 | 3859.90 | STOP_HIT | 0.50 | 2.85% |
